import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import einops
import numpy as np

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXEpic
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DatasetSegmentWrapper, DataManager, DataManagerCrossEval, DatasetWrapperEGTEA
# from dassl.data import DatasetSegmentWrapper, DataManager, Ego4DDataManager

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.dist_utils import get_rank, get_world_size

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None, bottle_neck: int=1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        hidden_dim = int(d_model / bottle_neck)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, hidden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_dim, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, bottle_neck=4):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask, bottle_neck) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # NOTE: this line is not yet completely clear

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, test_classnames=None, egtea_classes=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.cls_step = cfg.TRAINER.CLS_STEP
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # 16 x 512
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames] # NOTE: deleted "." in the end of sentence
        if cfg.TRAINER.COOP.WITHOUT_CLASSNAMES:
            prompts = [prompt_prefix + " " + '.' for name in classnames] # NOTE: deleted "." in the end of sentence

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        print("TOKEN embedding:", embedding.shape, flush=True)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        if test_classnames is not None:
            test_classnames = [name.replace("_", " ") for name in test_classnames]
            test_name_lens = [len(_tokenizer.encode(name)) for name in test_classnames]
            test_prompts = [prompt_prefix + " " + name for name in test_classnames] # NOTE: deleted "." in the end of sentence

            test_tokenized_prompts = torch.cat([clip.tokenize(p) for p in test_prompts])
            with torch.no_grad():
                test_embedding = clip_model.token_embedding(test_tokenized_prompts).type(dtype)

            print("TEST TOKEN embedding:", test_embedding.shape, flush=True)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("test_token_prefix", test_embedding[:, :1, :])  # SOS
            self.register_buffer("test_token_suffix", test_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
            self.test_tokenized_prompts = test_tokenized_prompts  # torch.Tensor


        if egtea_classes is not None:
            egtea_classes = [name.replace("_", " ") for name in egtea_classes]
            test_name_lens = [len(_tokenizer.encode(name)) for name in egtea_classes]
            test_prompts_egtea = [prompt_prefix + " " + name for name in egtea_classes] # NOTE: deleted "." in the end of sentence

            test_tokenized_promptsegtea = torch.cat([clip.tokenize(p) for p in test_prompts_egtea])
            with torch.no_grad():
                test_embeddingegtea = clip_model.token_embedding(test_tokenized_promptsegtea).type(dtype)

            print("TEST TOKEN EGTEA embedding:", test_embeddingegtea.shape, flush=True)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("test_token_prefix_egtea", test_embeddingegtea[:, :1, :])  # SOS
            self.register_buffer("test_token_suffix_egtea", test_embeddingegtea[:, 1 + n_ctx :, :])  # CLS, EOS
            self.test_tokenized_prompts_egtea = test_tokenized_promptsegtea  # torch.Tensor




        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, start=-1, end=-1, test=False, **kwargs):
        ctx = self.ctx
        step = self.cls_step
        if self.n_cls > step:
            assert start >= 0 and end > start
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(end-start, -1, -1)

            prefix = self.token_prefix[start:end]
            suffix = self.token_suffix[start:end]

            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                raise ValueError

            return prompts
        else:
            if test:
                prefix = self.test_token_prefix
                suffix = self.test_token_suffix
                n_cls = prefix.shape[0]
            elif 'test_egtea' in kwargs:
                # breakpoint()
                prefix = self.test_token_prefix_egtea
                suffix = self.test_token_suffix_egtea
                n_cls = prefix.shape[0]
            else:
                n_cls = self.n_cls
                prefix = self.token_prefix
                suffix = self.token_suffix

            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

            if self.class_token_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                raise ValueError

            return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device='cuda', test_classnames=None, egtea_classes=None):
        super().__init__()
        assert isinstance(classnames, dict) and len(classnames) == 1
        self.key = list(classnames.keys())[0]
        self.cfg = cfg
        classnames = classnames[self.key]
        test_classnames = test_classnames[self.key] if test_classnames is not None else test_classnames
        egtea_classes = egtea_classes[self.key] if egtea_classes is not None else egtea_classes
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, test_classnames=test_classnames, egtea_classes=egtea_classes)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.segments = cfg.DATALOADER.SEGMENTS
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.hidden_size = self.image_encoder.output_dim
        self.temporal = cfg.TRAINER.TEMPORAL.TYPE
        self.numF = cfg.DATALOADER.FRAMES_PER_SEGMENT
        self.device = device

        output_vis_ctx_dim = self.prompt_learner.ctx_dim
        self.use_dino_features = self.cfg.DATALOADER.USE_DINO_FEATURES


        if cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX:
            self.temporal_ctx = cfg.TRAINER.TEMPORAL.TYPE_CTX

            if cfg.TRAINER.DECOMP_COCOOP.SKIP_TEXT_ENCODER:
                output_vis_ctx_dim = self.image_encoder.output_dim

            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(self.hidden_size, self.hidden_size // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(self.hidden_size // 16, output_vis_ctx_dim)),
                ("relu2", nn.ReLU(inplace=True)),
            ]))

            if self.temporal_ctx not in ['max', 'avg']:
                self.temporalEmbedding_ctx = torch.nn.Embedding(self.numF, self.hidden_size)
                nn.init.normal_(self.temporalEmbedding_ctx.weight, std=0.01)

            # temporal part is borrowed from github.com/ju-chen/Efficient-Prompt
            if self.temporal_ctx in ['attention', 'attention_with_full_frames']:
                self.temporalModelling_ctx = TemporalModelling(
                    width=self.hidden_size,
                    layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                    heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                    dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                    bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )
            elif self.temporal_ctx in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
                # within the group
                self.temporalModelling_intra_frame_ctx = TemporalModelling(
                    width=self.hidden_size,
                    layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                    heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                    dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                    bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )
                # between the groups
                self.temporalModelling_inter_frame_ctx = TemporalModelling(
                    width=self.hidden_size,
                    layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS_TEMP,
                    heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                    dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                    bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )


        self.temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)

        # temporal part is borrowed from github.com/ju-chen/Efficient-Prompt
        if self.temporal in ['attention', 'attention_with_full_frames']:
            self.temporalModelling = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )
        elif self.temporal in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
            # within the group
            self.temporalModelling_intra_frame = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )
            # between the groups
            self.temporalModelling_inter_frame = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )


    def temporal_forward_ctx(self, visual_features, b, t):
        if self.segments:
            if self.temporal_ctx == 'attention':  # temporal modelling
                tempEmbedding = einops.repeat(self.temporalEmbedding_ctx(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = visual_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling_ctx(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal_ctx == 'attention_with_full_frames':
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding_ctx(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = visual_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1).view(b, num_extra_frames*t, -1)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling_ctx(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal_ctx in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
                num_extra_frames = 2
                t = t // num_extra_frames
                # breakpoint()
                # tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                # image_features = visual_features.view(b*num_extra_frames, t, -1)
                # image_features = image_features + tempEmbedding.to(self.device)
                image_features = visual_features.view(b, t, num_extra_frames, -1)  # b x extra x t x c
                image_features = image_features.view(b * t, num_extra_frames, -1)
                # image_features = visual_features.view(num_extra_frames, b, t, -1)  # extra x b x t x c
                image_features = image_features.transpose(0,1) # extra x (b * t) x c
                # image_features = image_features.reshape(num_extra_frames, b * t, -1) # extra x (b x t) x c
                image_features = self.temporalModelling_intra_frame_ctx(image_features)
                if 'avg' in self.temporal_ctx:
                    image_features = image_features.mean(dim=0)
                    image_features = image_features.view(b, t, -1) # (b x t) x c -> b x t x c
                    tempEmbedding = einops.repeat(self.temporalEmbedding_ctx(torch.arange(self.numF).to(self.device)),'t c -> b t c', b=b)
                    image_features = image_features.view(b, t, -1)
                    image_features = image_features + tempEmbedding.to(self.device)
                    image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                    image_features = self.temporalModelling_inter_frame_ctx(image_features)
                    image_features = image_features.mean(dim=0)
                elif 'reshape' in self.temporal_ctx:
                    image_features = image_features.view(num_extra_frames, b, t, -1) # extra x (b x t) x c -> extra x b x t x c
                    image_features = image_features.transpose(0,1) # extra x b x t x c  ->  b x extra x t x c
                    image_features = image_features.reshape(b, num_extra_frames * t, -1) # b x extra x t x c   -> b x (extra x t) x c
                    image_features = image_features.transpose(0,1) # b x (extra x t) x c  -> (extra x t) x b x c
                    image_features = self.temporalModelling_inter_frame_ctx(image_features)
                    image_features = image_features.mean(dim=0)
            elif self.temporal_ctx == 'avg':  # temporal modelling
                image_features = visual_features.view(b, t, -1).mean(dim=1)
            elif self.temporal_ctx == 'max':  # temporal modelling
                image_features = visual_features.view(b, t, -1).max(dim=1)[0]
            else:
                raise NotImplementedError('Check temporal function')
            if self.meta_net is not None:
                image_features = self.meta_net(image_features)
            image_features = image_features.unsqueeze(1)
        return image_features


    def forward(self, image, test=False, **kwargs):
        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX:
            if self.use_dino_features:
                ctx_image_features = image['dino']
                ctx_image_features = ctx_image_features / ctx_image_features.norm(dim=-1, keepdim=True)
                image = image['not_dino']
            else:
                ctx_image_features = image

        if self.segments:
            if self.use_extracted_features:
                assert len(image.shape) == 3
                b, t, c = image.shape
                image = image.view(-1, c)
            else:
                assert len(image.shape) == 5
                b, t, c, h, w = image.shape
                image = image.reshape(-1, c, h, w)

        if self.use_extracted_features:
            image_features = image.type(self.dtype)
        else:
            image_features = self.image_encoder(image.type(self.dtype))


        if self.segments:
            if self.temporal == 'attention':  # temporal modelling
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = image_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal == 'attention_with_full_frames':
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = image_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1).view(b, num_extra_frames*t, -1)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = image_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1)  # b x extra x t x c
                image_features = image_features.transpose(0,1) # extra x b x t x c
                image_features = image_features.reshape(num_extra_frames, b * t, -1) # extra x (b x t) x c
                image_features = self.temporalModelling_intra_frame(image_features)
                if 'avg' in self.temporal:
                    image_features = image_features.mean(dim=0)
                    image_features = image_features.view(b, t, -1) # (b x t) x c -> b x t x c
                    image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                    image_features = self.temporalModelling_inter_frame(image_features)
                    image_features = image_features.mean(dim=0)
                elif 'reshape' in self.temporal:
                    image_features = image_features.view(num_extra_frames, b, t, -1) # extra x (b x t) x c -> extra x b x t x c
                    image_features = image_features.transpose(0,1) # extra x b x t x c  ->  b x extra x t x c
                    image_features = image_features.reshape(b, num_extra_frames * t, -1) # b x extra x t x c   -> b x (extra x t) x c
                    image_features = image_features.transpose(0,1) # b x (extra x t) x c  -> (extra x t) x b x c
                    image_features = self.temporalModelling_inter_frame(image_features)
                    image_features = image_features.mean(dim=0)
            elif self.temporal == 'avg':  # temporal modelling
                image_features = image_features.view(b, t, -1).mean(dim=1)
            elif self.temporal == 'max':  # temporal modelling
                image_features = image_features.view(b, t, -1).max(dim=1)[0]
            else:
                raise NotImplementedError('Check temporal function')

        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX:
            b_dino, t_dino, c_dino = ctx_image_features.shape
            ctx_image_features = self.temporal_forward_ctx(ctx_image_features, b=b_dino, t=t_dino)


        if self.prompt_learner.n_cls > self.prompt_learner.cls_step:
            step = self.prompt_learner.cls_step
            n_cls = self.prompt_learner.n_cls
            # print('here', step, n_cls)
            text_features = None
            for start in range(0, n_cls+1, step):
                end = min(start + step, n_cls+1)
                # print('here2', start, end)
                prompts = self.prompt_learner(start, end)
                tokenized_prompts = self.tokenized_prompts[start:end]
                # print('here3', tokenized_prompts.shape, prompts.shape)
                text_features_tmp = self.text_encoder(prompts, tokenized_prompts)
                breakpoint()
                text_features = text_features_tmp if text_features is None else torch.cat((text_features, text_features_tmp))
        else:
            if test:
                tokenized_prompts = self.prompt_learner.test_tokenized_prompts
            else:
                tokenized_prompts = self.tokenized_prompts

            if 'test_egtea' in kwargs and self.prompt_learner.test_tokenized_prompts_egtea is not None:
                # breakpoint()
                tokenized_prompts = self.prompt_learner.test_tokenized_prompts_egtea

            prompts = self.prompt_learner(test=test, **kwargs)
            text_features = self.text_encoder(prompts, tokenized_prompts)

        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX:
            text_features = text_features.unsqueeze(0) + ctx_image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX:
            logits = logit_scale * ((image_features.unsqueeze(1) * text_features).sum(-1))
        else:
            logits = logit_scale * image_features @ text_features.t()

        return {self.key: logits}


@TRAINER_REGISTRY.register()
class TemporalCoOp(TrainerXEpic):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
        if 'Ego4D' in self.cfg.DATASET.NAME:
            from dassl.data import Ego4DDataManager
            dm = Ego4DDataManager(self.cfg)
        else:
            if self.cfg.DATALOADER.SEGMENTS:
                dm = DataManager(self.cfg, dataset_wrapper=DatasetSegmentWrapper)
            else:
                dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        print('NUM CLASSES', self.num_classes, flush=True)
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

        if self.cfg.TEST.CROSS_DATASET.EVAL:
            print(f'CROSS DATASET {self.cfg.TEST.CROSS_DATASET.DATASET_NAME}')
            self.cross_eval = True
            if 'Ego4D' in self.cfg.TEST.CROSS_DATASET.DATASET_NAME:
                from dassl.data import Ego4DDataManagerCrossEval
                self.cross_dm = Ego4DDataManagerCrossEval(self.cfg)
            else:
                self.cross_dm = DataManagerCrossEval(self.cfg, dataset_wrapper=DatasetSegmentWrapper)

            self.cross_val_loader = self.cross_dm.val_loader
            self.cross_test_loader = self.cross_dm.test_loader
            self.cross_num_classes = self.cross_dm.num_classes
            print('CROSS NUM CLASSES', self.cross_num_classes, flush=True)

            if self.cfg.TEST.CROSS_DATASET.RETRIEVAL:
                self.cross_text_val_dataloader = self.cross_dm.text_val_dataloader

        if self.cfg.TEST.CROSS_DATASET.EGTEA:
            print('CROSS DATASET EGTEA')
            self.cross_eval_egtea = True
            self.cross_dm_egtea = DataManagerCrossEval(self.cfg, dataset_wrapper=DatasetWrapperEGTEA, egtea=True)
            self.cross_val_egtea_loader = self.cross_dm_egtea.val_loader
            self.cross_test_egtea_loader = self.cross_dm_egtea.test_loader
            self.cross_egtea_num_classes = self.cross_dm_egtea.num_classes
            print('CROSS NUM CLASSES EGTEA', self.cross_egtea_num_classes, flush=True)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        # print('BUILD MODEL', classnames, flush=True)
        # print('BUILD MODEL', len(classnames), flush=True)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if self.cross_eval_egtea:
            egtea_classes = self.cross_dm_egtea.dataset.classnames
        else:
            egtea_classes = None

        print("Building custom CLIP")
        if (cfg.DATASET.LABEL_TYPE == 'noun' and cfg.DATASET.SUBSET == 'seen_nouns') or \
            (cfg.DATASET.LABEL_TYPE == 'action' and cfg.DATASET.SUBSET == 'ek-55'):
            test_classnames = self.dm.dataset.test_classes
            self.model = CustomCLIP(cfg, classnames, clip_model, test_classnames=test_classnames, egtea_classes=egtea_classes)
        else:
            if self.cross_eval:
                test_classnames = self.cross_dm.dataset.classnames
                self.model = CustomCLIP(cfg, classnames, clip_model, test_classnames=test_classnames, egtea_classes=egtea_classes)
            else:
                self.model = CustomCLIP(cfg, classnames, clip_model, egtea_classes=egtea_classes)
            # self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and 'temporal' not in name and 'logit_scale' not in name and 'meta_net' not in name:
                param.requires_grad_(False)
            else:
                print(f'Params with grad: {name}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        print('Device', self.device)
        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        if cfg.DISTRIBUTED:
            print(f"Multiple GPUs detected (n_gpus={get_world_size()}), use all of them!", flush=True)
            # print('Apply SyncBN', flush=True)
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            local_rank = int(os.environ["LOCAL_RANK"])
            self.local_rank = local_rank
            # torch.cuda.set_device(local_rank)
            print(f'Apply DDP {local_rank}', flush=True)
            self.model.cuda(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
            print(f'Done DDP {local_rank}')
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model


    def model_inference(self, input, test=False, **kwargs):
        return self.model_without_ddp(input, test=test, **kwargs)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                # print('OUTPUT', output, flush=True)
                # print('MAX output', output.max(dim=-1), flush=True)
                # print('output.shape', output.shape, flush=True)
                # print('LABEL', label, flush=True)
                loss = None
                for label_type in output.keys():
                    tmp_loss = F.cross_entropy(output[label_type], label[label_type])
                    if loss is None: loss = tmp_loss
                    else: loss += tmp_loss

                # loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            # print('OUTPUT', output, flush=True)
            # print('MAX output', output.max(dim=-1), flush=True)
            # print('output.shapes', output.shape, flush=True)
            # print('LABEL', label, flush=True)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, label)[0].item(),
        }
        for label_type in output.keys():
            loss_summary.update({
                f"acc_{label_type}": compute_accuracy(output[label_type], label[label_type])[0].item(),
            })

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch):
        if 'img' in batch:
            input = batch["img"]
            if 'label' in batch:
                label = batch["label"]
                label = label.to(self.device)
            else:
                label = {}
                for label_type in ['noun', 'verb', 'action']:
                    label_tmp = batch[f'{label_type}_label']
                    label[label_type] = label_tmp.to(self.device)

            input = input.to(self.device)
            return input, label
        else:
            # ego4d dataset
            # breakpoint()
            input = batch[0][0].transpose(1,2)
            #x["verb_label"], x["noun_label"]
            if self.cfg.DATASET.LABEL_TYPE == 'noun':
                label = batch[1][:, 1]
            elif self.cfg.DATASET.LABEL_TYPE == 'verb':
                label = batch[1][:, 0]
            input = input.to(self.device)
            label = label.to(self.device)
            return input, label

    def parse_batch_test(self, batch):
        if 'img' in batch:
            input = batch["img"]
            if 'label' in batch:
                label = batch["label"]
                label = label.to(self.device)
            else:
                label = {}
                for label_type in ['noun', 'verb', 'action']:
                    label_tmp = batch[f'{label_type}_label']
                    label[label_type] = label_tmp.to(self.device)
            narration_id = batch['narration_id']

            input = input.to(self.device)


            return input, label, narration_id
        else:
            # ego4d dataset
            # breakpoint()
            input = batch[0][0].transpose(1, 2)
            # x["verb_label"], x["noun_label"]
            if self.cfg.DATASET.LABEL_TYPE == 'noun':
                label = batch[1][:, 1]
            elif self.cfg.DATASET.LABEL_TYPE == 'verb':
                label = batch[1][:, 0]
            input = input.to(self.device)
            label = label.to(self.device)
            return input, label, None


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.test_token_prefix" in state_dict:
                del state_dict["prompt_learner.test_token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            if "prompt_learner.test_token_suffix" in state_dict:
                del state_dict["prompt_learner.test_token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
