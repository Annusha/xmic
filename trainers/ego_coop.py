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
from dassl.data import DatasetSegmentWrapper, DataManager

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
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
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
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

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
        # print('X: prompt+ pos', x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # import pdb;pdb.set_trace()
        # breakpoint()
        x = self.transformer(x)
        # print('X: transformer', x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print('X: ln_final', x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # NOTE: this line is not yet completely clear

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, subclassnames=None, test_classnames=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
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
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if cfg.TRAINER.EGO_COOP.SUBCLASSNAMES_SC:
            subclass_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(subclass_ctx_vectors, std=0.02)
            self.subclass_ctx = nn.Parameter(subclass_ctx_vectors)
        else:
            self.subclass_ctx = self.ctx

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames] # NOTE: deleted "." in the end of sentence

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


        if cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
            # import pdb;pdb.set_trace()
            # breakpoint()
            subclassnames = [name.replace("_", " ") for name in subclassnames]
            subclass_name_lens = [len(_tokenizer.encode(name)) for name in subclassnames]
            subclass_prompts = [prompt_prefix + " " + name for name in subclassnames] # NOTE: deleted "." in the end of sentence
            # print('subclass_prompts', subclass_prompts)

            subclass_tokenized_prompts = torch.cat([clip.tokenize(p) for p in subclass_prompts])
            with torch.no_grad():
                subclass_embedding = clip_model.token_embedding(subclass_tokenized_prompts).type(dtype)

            print("TOKEN subclass embedding:", subclass_embedding.shape, flush=True)
            print("TOKEN subclass embedding sum:", subclass_embedding.sum(), flush=True)

            self.register_buffer("subclass_token_prefix", subclass_embedding[:, :1, :])  # SOS
            self.register_buffer("subclass_token_suffix", subclass_embedding[:, 1 + n_ctx :, :])  # CLS, EOS

            self.subclass_n_cls = cfg.TRAINER.EGO_COOP.N_SUBCLASSES
            self.subclass_tokenized_prompts = subclass_tokenized_prompts  # torch.Tensor
            self.subclass_name_lens = subclass_name_lens
            self.subclass_indexes = set(range(len(subclassnames)))

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    # def create_test_prompts(self, cfg, clip_model, test_classnames):
    #     test_classnames = [name.replace("_", " ") for name in test_classnames]
    #     test_name_lens = [len(_tokenizer.encode(name)) for name in test_classnames]
    #     test_prompts = [prompt_prefix + " " + name for name in test_classnames] # NOTE: deleted "." in the end of sentence

    #     test_tokenized_prompts = torch.cat([clip.tokenize(p) for p in test_prompts])
    #     with torch.no_grad():
    #         test_embedding = clip_model.token_embedding(test_tokenized_prompts).type(dtype)

    #     print("TEST TOKEN embedding:", test_embedding.shape, flush=True)

    #     # These token vectors will be saved when in save_model(),
    #     # but they should be ignored in load_model() as we want to use
    #     # those computed using the current class names
    #     self.register_buffer("test_token_prefix", test_embedding[:, :1, :])  # SOS
    #     self.register_buffer("test_token_suffix", test_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
    #     self.test_tokenized_prompts = test_tokenized_prompts  # torch.Tensor


    def forward(self, subclass_indexes=None, test=False):
        # breakpoint()


        if test:
            prefix = self.test_token_prefix
            suffix = self.test_token_suffix
            n_cls = prefix.shape[0]
        else:
            n_cls = self.n_cls
            prefix = self.token_prefix
            suffix = self.token_suffix

        ctx = self.ctx
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

        # breakpoint()
        if subclass_indexes is not None:
            # print('subclass_ctx', self.subclass_ctx.sum())
            subclass_ctx = self.subclass_ctx
            if subclass_ctx.dim() == 2:
                subclass_ctx = subclass_ctx.unsqueeze(0).expand(self.subclass_n_cls, -1, -1)
                # print('subclass_ctx 2', subclass_ctx.sum())


            free_indexes = self.subclass_indexes - set(subclass_indexes)
            additional_indexes = np.random.choice(list(free_indexes), self.subclass_n_cls - len(subclass_indexes))
            subclass_indexes = np.concatenate([subclass_indexes, np.array(additional_indexes)])
            # print(subclass_indexes)
            # print(subclass_indexes.shape, subclass_indexes.max(), subclass_indexes.min())
            # print(self.token_prefix.shape, self.token_suffix.shape)
            prefix = self.subclass_token_prefix[subclass_indexes]
            suffix = self.subclass_token_suffix[subclass_indexes]
            # print('prefix', prefix.sum() )
            # print('suffix', suffix.sum() )

            # import pdb;pdb.set_trace()
            # breakpoint()
            subclass_prompts = torch.cat([prefix, subclass_ctx, suffix], dim=1 )


            return prompts, subclass_prompts, subclass_indexes

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, device='cuda', subclassnames=None, test_classnames=None, classnames_detic=None):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, subclassnames, test_classnames)
        # import pdb;pdb.set_trace()
        # breakpoint()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        if cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
            self.subclass_tokenized_prompts = self.prompt_learner.subclass_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.segments = cfg.DATALOADER.SEGMENTS
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.use_objects_features = cfg.DATALOADER.USE_OBJECTS_FEATURES
        self.hidden_size = self.image_encoder.output_dim
        self.temporal = cfg.TRAINER.TEMPORAL.TYPE
        self.numF = cfg.DATALOADER.FRAMES_PER_SEGMENT
        self.device = device
        self.cfg = cfg
        self.T = cfg.TRAINER.EGO_COOP.TEMPERATURE

        if self.use_objects_features and cfg.TRAINER.EGO_COOP.OBJECTS.USE_BCE:
            self.class_embeddings = self._init_class_prompts(classnames, clip_model)

        if self.use_objects_features and cfg.DATALOADER.DETIC.USE_BCE:
            self.class_embeddings_detic = self._init_class_prompts(classnames_detic, clip_model)


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


    def _init_class_prompts(self, classnames, clip_model):
        dtype = clip_model.dtype
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = ["a photo of a " + name for name in classnames] # NOTE: deleted "." in the end of sentence

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            text_features = self.text_encoder(embedding, tokenized_prompts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def forward(self, image, subclass_indexes=None, use_img_obj=False, temporal=None, test=False, num_extra_frames=2, temp_avg=True, return_image_features=False):
        # import pdb;pdb.set_trace()
        # breakpoint()

        if self.segments:
            if self.use_extracted_features:
                if use_img_obj:
                    if 'detic_objs' in image:
                        detic_objs = image['detic_objs']
                        det_b, det_t, det_objs, det_c = detic_objs.shape
                        image = image['img']
                        b, t, c = image.shape

                        image = torch.cat([image.unsqueeze(2), detic_objs], dim=2)
                        image = image.reshape(b, -1, c)

                    elif 'img_obj' in image:
                        obj_feat = image['img_obj']
                        image = image['img']
                        b, t, c = image.shape

                        image = torch.cat([image.unsqueeze(0), obj_feat.unsqueeze(0)])
                        image = image.transpose(0,1)
                        image = image.transpose(1,2)
                        image = image.reshape(b, -1, c)

                # breakpoint()
                if len(image.shape) == 4:
                    b,t,n,c = image.shape
                    t = t * n
                    assert n == num_extra_frames
                    image = image.view(-1, c)
                else:
                    assert len(image.shape) == 3
                    b, t, c = image.shape
                    image = image.view(-1, c)
            else:
                assert len(image.shape) == 5
                b, t, c, h, w = image.shape
                image = image.view(-1, c, h, w)

        if self.use_extracted_features:
            image_features = image.type(self.dtype)
        else:
            image_features = self.image_encoder(image.type(self.dtype))


        if self.segments:
            if self.temporal == 'attention' or temporal == 'attention':  # temporal modelling
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = image_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                if temp_avg:
                    image_features = image_features.mean(dim=0)
                else:
                    image_features = image_features.transpose(0,1) # .view(b*t, -1)
            elif self.temporal == 'attention_with_full_frames':
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = image_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1).view(b, num_extra_frames*t, -1)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                if temp_avg:
                    image_features = image_features.mean(dim=0)
                else:
                    image_features = image_features.transpose(0,1) # .view(b*num_extra_frames*t, -1)
            elif self.temporal in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
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
                    if temp_avg:
                        image_features = image_features.mean(dim=0)
                    else:
                        image_features = image_features.transpose(0,1) # .view(b*t, -1)
                elif 'reshape' in self.temporal:
                    image_features = image_features.view(num_extra_frames, b, t, -1) # extra x (b x t) x c -> extra x b x t x c
                    image_features = image_features.transpose(0,1) # extra x b x t x c  ->  b x extra x t x c
                    image_features = image_features.reshape(b, num_extra_frames * t, -1) # b x extra x t x c   -> b x (extra x t) x c
                    image_features = image_features.transpose(0,1) # b x (extra x t) x c  -> (extra x t) x b x c
                    image_features = self.temporalModelling_inter_frame(image_features)
                    if temp_avg:
                        image_features = image_features.mean(dim=0)
                    else:
                        image_features = image_features.transpose(0,1) # .view(b*num_extra_frames*t, -1)
            elif self.temporal == 'avg':  # temporal modelling
                image_features = image_features.view(b, t, -1).mean(dim=1)
            else:
                raise NotImplementedError('Check temporal function')

        # breakpoint()
        output = {}

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if return_image_features:
            output.update({
                'image_features': image_features,
            })

            return output


        if self.cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES and subclass_indexes is not None:
            prompts, subclass_prompts, subclass_indexes = self.prompt_learner(subclass_indexes=subclass_indexes, test=test)
        else:
            prompts = self.prompt_learner(test=test)

        if test:
            tokenized_prompts = self.prompt_learner.test_tokenized_prompts
        else:
            tokenized_prompts = self.tokenized_prompts
        # print('tokenzied_prompts', tokenized_prompts)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # if not temp_avg:
        #     b_tmp, c_tmp = logits.shape
        #     logits = logits.view(b, -1, c_tmp)


        output.update({'logits': logits})


        if self.cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES and subclass_indexes is not None:
            # breakpoint()
            # print('subclass_indexes', subclass_indexes)

            subclass_tokenized_prompts = self.subclass_tokenized_prompts[subclass_indexes]
            subclass_text_features = self.text_encoder(subclass_prompts, subclass_tokenized_prompts)
            # print('subclass_tokenized_prompts', subclass_tokenized_prompts)
            # print('subclass_text_features', subclass_text_features)

            subclass_text_features = subclass_text_features / subclass_text_features.norm(dim=-1, keepdim=True)

            # print('norm subclass_text_features', subclass_text_features)

            if self.T > 0:
                subclass_logits = image_features @ subclass_text_features.t()
                subclass_logits /= self.T
            else:
                subclass_logits = logit_scale * image_features @ subclass_text_features.t()
            output.update({'subclass_logits': subclass_logits})

        return output


@TRAINER_REGISTRY.register()
class EgoCoOp(TrainerXEpic):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):
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

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        if cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
            subclassnames = self.dm.dataset.subclassnames
            print('BUILD MODEL', len(subclassnames), flush=True)
        else:
            subclassnames = None
            print('BUILD MODEL', 'len(subclassnames)', flush=True)
        # print('BUILD MODEL', classnames, flush=True)
        # print('BUILD MODEL', len(classnames), flush=True)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        classnames_detic = None
        if cfg.DATALOADER.DETIC.USE_BCE:
            classnames_detic = self.dm.dataset.classnames_detic

        if cfg.DATASET.LABEL_TYPE == 'noun' and cfg.DATASET.SUBSET == 'seen_nouns':
            test_classnames = self.dm.dataset.test_classes
            self.model = CustomCLIP(cfg, classnames, clip_model, subclassnames=subclassnames, test_classnames=test_classnames, classnames_detic=classnames_detic)
        else:
            self.model = CustomCLIP(cfg, classnames, clip_model, subclassnames=subclassnames, classnames_detic=classnames_detic)
        self.use_img_obj = cfg.TRAINER.EGO_COOP.OBJECTS.CONCAT_IMG_AND_OBJS
        self.img_obj_temp_avg = cfg.TRAINER.EGO_COOP.OBJECTS.IMG_AND_OBJS_TEMP_AVG or cfg.DATALOADER.DETIC.IMG_AND_OBJS_TEMP_AVG
        # breakpoint()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and 'temporal' not in name:
                param.requires_grad_(False)
            else:
                print(f'Params with grad: {name}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        print('Device', self.device)
        self.model.to(self.device)
        if self.model.use_objects_features and cfg.DATALOADER.DETIC.USE_BCE:
            self.model.class_embeddings_detic = self.model.class_embeddings_detic.to(self.device)

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


    def model_inference(self, input, test=False):
        if self.cfg.DATALOADER.DETIC.CROPS:
            return self.model_without_ddp(input, use_img_obj=self.use_img_obj, num_extra_frames=self.cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME + 1, test=test)['logits']
        else:
            return self.model_without_ddp(input, use_img_obj=self.use_img_obj, test=test)['logits']

    def forward_backward(self, batch):
        input_dict = self.parse_batch_train(batch)
        image = input_dict['input']
        label = input_dict['label']
        subclass_indexes = input_dict['subclass_idx']

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                if self.cfg.DATALOADER.DETIC.CROPS:
                    output = self.model(image, subclass_indexes=subclass_indexes, use_img_obj=self.use_img_obj, num_extra_frames=self.cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME + 1)
                else:
                    output = self.model(image, subclass_indexes=subclass_indexes, use_img_obj=self.use_img_obj)
                # print('OUTPUT', output, flush=True)
                # print('MAX output', output.max(dim=-1), flush=True)
                # print('output.shape', output.shape, flush=True)
                # print('LABEL', label, flush=True)
                loss = F.cross_entropy(output['logits'], label)
                if self.cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
                    subclass_labels = torch.arange(output['subclass_logits'].shape[0], dtype=torch.long).to(self.device)
                    # print(output['subclass_logits'])
                    # print(output['subclass_logits'].shape)
                    # print(subclass_labels)
                    # print('Loss before', loss)
                    if self.cfg.TRAINER.EGO_COOP.WITHOUT_CLASS_LOSS:
                        loss = self.cfg.TRAINER.EGO_COOP.SUBCLASS_LOSS_WEIGHT * F.cross_entropy(output['subclass_logits'], subclass_labels)
                    else:
                        loss += self.cfg.TRAINER.EGO_COOP.SUBCLASS_LOSS_WEIGHT * F.cross_entropy(output['subclass_logits'], subclass_labels)
                    # print('Loss after', loss_subcls)
                    # loss += loss_subcls
                    # exit(0)
                if self.cfg.TRAINER.EGO_COOP.OBJECTS.USE_BCE:
                    obj_labels = input_dict['obj_labels']
                    output_obj = self.model(image['img_obj'], subclass_indexes=subclass_indexes, temporal='attention', temp_avg=self.img_obj_temp_avg)

                    # loss to have all predicted objects on average in the segment
                    if self.cfg.TRAINER.EGO_COOP.OBJECTS.SEGMENT_LEVEL_BCE:
                        logits = output_obj['logits'] if self.img_obj_temp_avg else output_obj['logits'].sum(1)
                        loss += self.cfg.TRAINER.EGO_COOP.OBJECTS.BCE_WEIGHT * F.binary_cross_entropy_with_logits(logits, obj_labels)

                    # loss to have objects in the individual frames
                    if self.cfg.TRAINER.EGO_COOP.OBJECTS.FRAME_LEVEL_BCE:
                        assert not self.img_obj_temp_avg
                        # breakpoint()
                        dim = obj_labels.shape[-1]
                        obj_labels = obj_labels.unsqueeze(1).repeat(1, output_obj['logits'].shape[1], 1)
                        with torch.no_grad():
                            # make segment labels more frame-oriented, do not penalize missed predictions on the frame basis
                            # as the ground truth is still on the segment level
                            # I'll try frame-level gt as well
                            obj_labels_mask = ((obj_labels * 0.7) + (output_obj['logits'] > 0)) < 1
                            obj_labels[obj_labels_mask] *= 0
                        obj_labels = obj_labels.view(-1, dim)
                        logits = output_obj['logits'].view(-1, dim)
                        loss += self.cfg.TRAINER.EGO_COOP.OBJECTS.FRAME_LEVEL_BCE_WEIGHT * F.binary_cross_entropy_with_logits(logits, obj_labels)

                if self.cfg.DATALOADER.DETIC.USE_BCE:
                    obj_labels = input_dict['detic_labels']
                    self.model = self.model.to(self.device)
                    output_obj = self.model(image['detic_objs'], subclass_indexes=subclass_indexes, temp_avg=self.img_obj_temp_avg, num_extra_frames=self.cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME,
                                            return_image_features=True)

                    image_features = output_obj['image_features']
                    logit_scale = self.model.logit_scale.exp()
                    # self.model.class_embeddings_detic = self.model.class_embeddings_detic.to(self.device)
                    logits = logit_scale * image_features @ self.model.class_embeddings_detic.t()
                    # loss to have all predicted objects on average in the segment
                    if self.cfg.DATALOADER.DETIC.SEGMENT_LEVEL_BCE:
                        logits = logits if self.img_obj_temp_avg else logits.sum(1)
                        obj_labels_segment_level = (obj_labels.sum(1) >= 0).type(torch.float32).to(self.device)
                        loss += self.cfg.DATALOADER.DETIC.SEGMENT_LEVEL_BCE_WEIGHT * F.binary_cross_entropy_with_logits(logits, obj_labels_segment_level)

                    # loss to have objects in the individual frames
                    if self.cfg.DATALOADER.DETIC.FRAME_LEVEL_BCE:
                        assert not self.img_obj_temp_avg
                        obj_labels = obj_labels.view(-1, dim)
                        logits = logits.view(-1, dim)
                        loss += self.cfg.DATALOADER.DETIC.FRAME_LEVEL_BCE_WEIGHT * F.binary_cross_entropy_with_logits(logits, obj_labels)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.use_img_obj:
                raise Exception('Not Implemented')
            output = self.model(image, subclass_indexes=subclass_indexes)
            # print('OUTPUT', output, flush=True)
            # print('MAX output', output.max(dim=-1), flush=True)
            # print('output.shapes', output.shape, flush=True)
            # print('LABEL', label, flush=True)
            loss = F.cross_entropy(output['logits'], label)
            if self.cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
                subclass_labels = torch.arange(output['subclass_logits'].shape[0], dtype=torch.long).to(self.device)
                if self.cfg.TRAINER.EGO_COOP.WITHOUT_CLASS_LOSS:
                    loss = F.cross_entropy(output['subclass_logits'], subclass_labels)
                else:
                    loss += F.cross_entropy(output['subclass_logits'], subclass_labels)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output['logits'], label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        output = {'input':input, 'label': label}
        if self.cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES:
            # print(batch["subclass_label"])
            subclass_label = batch["subclass_label"].numpy().reshape(-1)
            output.update({'subclass_idx': subclass_label})
        else:
            output.update({'subclass_idx': None})

        if self.cfg.DATALOADER.USE_OBJECTS_FEATURES:
            if self.cfg.DATALOADER.DETIC.CROPS:
                input_objects = batch['detic_objs']
                input_objects = input_objects.to(self.device)
                tmp_input = {'img': input, 'detic_objs': input_objects}
                output.update({'input': tmp_input})
                detic_labels = (batch['detic_labels'].sum(2) > 0).type(torch.float32)  # labels in the form b x t x n_obj_per_frame x n_classes; we make them frame level labels
                detic_labels = detic_labels.to(self.device)
                output.update({'detic_labels': detic_labels})
            else:
                input_objects = batch['img_obj']
                input_objects = input_objects.to(self.device)
                tmp_input = {'img': input, 'img_obj': input_objects}
                output.update({'input': tmp_input})

            if self.cfg.TRAINER.EGO_COOP.OBJECTS.USE_BCE:

                input_objects2 = input_objects / input_objects.norm(dim=-1, keepdim=True)
                self.model.class_embeddings = self.model.class_embeddings.to(self.device)
                logits = input_objects2 @ self.model.class_embeddings.t()

                labels = ((logits > self.cfg.TRAINER.EGO_COOP.OBJECTS.BCE_THD).sum(1) > 0).type(torch.float32)
                output.update({'obj_labels': labels})

        return output

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        narration_id = batch['narration_id']

        input = input.to(self.device)
        label = label.to(self.device)

        if self.cfg.DATALOADER.USE_OBJECTS_FEATURES:
            if self.cfg.DATALOADER.DETIC.CROPS:
                input_objects = batch['detic_objs']
                input_objects = input_objects.to(self.device)
                input = {'img': input, 'detic_objs': input_objects}
                # output.update({'input': tmp_input})
                # detic_labels = (batch['detic_labels'].sum(2) > 0).type(torch.float32)  # labels in the form b x t x n_obj_per_frame x n_classes; we make them frame level labels
                # output.update({'detic_labels': detic_labels})
            else:
                input_objects = batch['img_obj']
                input_objects = input_objects.to(self.device)
                input = {'img': input, 'img_obj': input_objects}
            # output.update{'input': tmp_input}

        return input, label, narration_id


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

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
