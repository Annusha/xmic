import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import einops
import numpy as np
#

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXEpic
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DatasetSegmentWrapper, DataManager, DataManagerCrossEval, DatasetWrapperEGTEA
# from dassl.data import DatasetSegmentWrapper, DataManager, Ego4DDataManager

from lavila import models
from lavila.tokenizer import SimpleTokenizer
from lavila.utils import inflate_positional_embeds

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.dist_utils import get_rank, get_world_size

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "EpicKitchen": "a photo of a {}.",
    "EpicKitchenSegments": "a photo of a {}",
    # "EpicKitchenSegments": "a photo of hands holding a {}",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    # "Ego4DRecognitionWrapper": "a photo of a {}"
    "Ego4DRecognitionWrapper": "a photo of a {}"
}
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
    def __init__(self, clip_model, framework='clip'):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        if framework == 'clip':
            self.dtype = clip_model.dtype
        elif framework == 'lavila':
            self.dtype = clip_model.visual.blocks[0].attn.qkv.weight.dtype

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
    def __init__(self, cfg, clip_model):
        super().__init__()
        if cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
            dtype = clip_model.dtype
        elif cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
            dtype = clip_model.visual.blocks[0].attn.qkv.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        self.cls_step = cfg.TRAINER.CLS_STEP
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.visual_with_ctx = cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX
        # self.token_embedding = clip_model.token_embedding  # add deletion in the upload of the model
        self.scale_factor_vis_ctx = cfg.TRAINER.DECOMP_COCOOP.FIXED_SCALE_FACTOR_VIS_CTX  # 1.0 default
        if cfg.TRAINER.DECOMP_COCOOP.LEARNABLE_SCALE_FACTOR_VIS_CTX:
            self.scale_factor_vis_ctx = nn.Parameter(data=torch.Tensor([self.scale_factor_vis_ctx]), requires_grad=True)


        self.scale_factor_v2t_narrations = cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_V2T  # 1.0 default
        self.scale_factor_t2v_narrations = cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_T2V  # 1.0 default
        if 'narration' in cfg.DATASET.LABEL_SUBTYPES:
            if cfg.TRAINER.DECOMP_COCOOP.LEARNABLE_SCALE_FACTOR_VIS_CTX:
                self.scale_factor_v2t_narrations = nn.Parameter(data=torch.Tensor([self.scale_factor_v2t_narrations]), requires_grad=True)
                self.scale_factor_t2v_narrations = nn.Parameter(data=torch.Tensor([self.scale_factor_t2v_narrations]), requires_grad=True)

        # visual tokens
        if self.visual_with_ctx:
            n_ctx_visual_learnable = cfg.TRAINER.DECOMP_COCOOP.VISUAL_N_CTX

            prompt_prefix_vis = " ".join(["X"] * n_ctx_visual_learnable)
            tokenized_visual_prompt = clip.tokenize(prompt_prefix_vis + " V")
            n_visual_ctx = n_ctx_visual_learnable + 1
            assert cfg.TRAINER.DECOMP_COCOOP.VISUAL_CTX_NEW
            print("Initializing a generic context")
            ctx_vectors_visual = torch.empty(n_ctx_visual_learnable, ctx_dim, dtype=dtype)  # 16 x 512
            nn.init.normal_(ctx_vectors_visual, std=0.02)
            self.ctx_visual = nn.Parameter(ctx_vectors_visual)

        else:
            tokenized_visual_prompt = clip.tokenize("V")
            n_visual_ctx = 1

        with torch.no_grad():
            visual_embedding = clip_model.token_embedding(tokenized_visual_prompt).type(dtype)

        print("TOKEN visual embedding:", visual_embedding.shape, flush=True)

        self.register_buffer("token_visual_prefix", visual_embedding[:, :1, :])  # SOS
        self.register_buffer("token_visual_suffix", visual_embedding[:, 1+n_visual_ctx:, :])  # CLS, EOS
        self.tokenized_visual_prompts = tokenized_visual_prompt
        self.n_visual_ctx = n_visual_ctx

        self.dtype = dtype



    def forward(self, ctx_vis_features=None):

        # visual prompts
        batch_size = ctx_vis_features.shape[0]

        prefix = self.token_visual_prefix
        prefix = prefix.expand(batch_size, -1, -1)

        suffix = self.token_visual_suffix
        suffix = suffix.expand(batch_size, -1, -1)

        if self.visual_with_ctx:
            ctx = self.ctx_visual if self.ctx_visual is not None else self.ctx
            ctx = ctx.unsqueeze(0).expand(batch_size, -1, -1)
            visual_prompts = torch.cat([prefix, ctx, ctx_vis_features, suffix], dim=1)
        else:
            visual_prompts = torch.cat([prefix, ctx_vis_features, suffix], dim=1)

        # ( n_cls, 77, dim )  (b, 77, dim)
        return visual_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_sets, clip_model, device='cuda', test_class_sets=None, egtea_classes=None):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, clip_model)
        self.prompt_prefix = cfg.TRAINER.DECOMP_COCOOP.PROMPT_PREFIX
        self.tokenized_visual_prompts = self.prompt_learner.tokenized_visual_prompts
        # TODO:  need to add image_projector for lavila, see lavila_ft!!!
        output_vis_ctx_dim = self.prompt_learner.ctx_dim
        self.image_encoder = clip_model.visual
        if cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
            self.dtype = clip_model.dtype
            self.hidden_size = self.image_encoder.output_dim
            if cfg.TRAINER.DECOMP_COCOOP.SKIP_TEXT_ENCODER:
                output_vis_ctx_dim = self.image_encoder.output_dim
            output_dim_narr = self.image_encoder.output_dim
            input_dim_narr = self.image_encoder.output_dim
        elif cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
            self.image_projection = clip_model.image_projection
            self.dtype = clip_model.visual.blocks[0].attn.qkv.weight.dtype
            self.hidden_size = self.image_encoder.embed_dim
            if cfg.TRAINER.DECOMP_COCOOP.SKIP_TEXT_ENCODER:
                output_vis_ctx_dim = self.image_projection.shape[1]
            output_dim_narr = self.image_projection.shape[1]
            input_dim_narr = self.image_projection.shape[1]
        self.text_encoder = TextEncoder(clip_model, framework=cfg.MODEL.BACKBONE.FRAMEWORK)
        self.logit_scale = clip_model.logit_scale # I might need to move it to learnable part
        self.segments = cfg.DATALOADER.SEGMENTS
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.use_dino_features = cfg.DATALOADER.USE_DINO_FEATURES
        if self.use_dino_features:
            self.hidden_size = cfg.DATALOADER.DINO_DIM
        self.temporal = cfg.TRAINER.TEMPORAL.TYPE
        self.temporal_backbone = cfg.TRAINER.TEMPORAL.BACKBONE_TYPE
        self.numF = cfg.DATALOADER.FRAMES_PER_SEGMENT
        self.with_relu = cfg.TRAINER.DECOMP_COCOOP.WITH_RELU
        self.device = device
        if cfg.DATALOADER.USE_EXTRACTED_FEATURES:
            self.image_encoder = None
        else:
            self.image_encoder = clip_model.visual

        print('Hidden dim', self.hidden_size)
        print('output_vis_ctx_dim', output_vis_ctx_dim)
        print('output_dim_narr', output_dim_narr)
        print('input_dim_narr', input_dim_narr)

        self.text_conditioning = cfg.TRAINER.DECOMP_COCOOP.TEXT_CONDITIONING
        self.vis_conditioning = cfg.TRAINER.DECOMP_COCOOP.VIS_CONDITIONING
        self.clip_adapter_ratio = cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER_RATIO

        self.token_embedding = clip_model.token_embedding
        tmp_root = f'../data/text_embeddings/{cfg.DATASET.NAME}/'
        os.makedirs(tmp_root, exist_ok=True)
        self.class_embeddings = self.class_sets_forward(class_sets, root=tmp_root)
        if test_class_sets is not None:
            tmp_root = f'../data/text_embeddings/{cfg.TEST.CROSS_DATASET.DATASET_NAME}'
            os.makedirs(tmp_root, exist_ok=True)
            self.test_class_embeddings = self.class_sets_forward(test_class_sets, root=tmp_root)

        if egtea_classes is not None:
            tmp_root = f'../data/text_embeddings/EGTEA/'
            os.makedirs(tmp_root, exist_ok=True)
            self.test_class_embeddings_egtea = self.class_sets_forward(egtea_classes, root=tmp_root)


        if cfg.TRAINER.DECOMP_COCOOP.MLP_LY == 2:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(self.hidden_size, self.hidden_size // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(self.hidden_size // 16, output_vis_ctx_dim))
            ]))
            if self.with_relu:
                self.meta_net = nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(self.hidden_size, self.hidden_size // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(self.hidden_size // 16, output_vis_ctx_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                ]))

        elif cfg.TRAINER.DECOMP_COCOOP.MLP_LY == 1:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(self.hidden_size, output_vis_ctx_dim)),
            ]))
            if self.with_relu:
                self.meta_net = nn.Sequential(OrderedDict([
                    ("linear1", nn.Linear(self.hidden_size, output_vis_ctx_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                ]))
        elif cfg.TRAINER.DECOMP_COCOOP.MLP_LY == 0:
            self.meta_net = None
        else:
            raise NotImplementedError('MLP can be only with 1 or 2 layers')


        #############################################
        ##### start ####  CLIP ADAPTER  ############
        if cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(self.hidden_size, self.hidden_size // 4, bias=False)),
                ("relu1", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(self.hidden_size // 4, output_dim_narr, bias=False)),
                ("relu2", nn.ReLU(inplace=True))
            ]))

            self.meta_net_txt2txt = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(input_dim_narr, self.hidden_size // 4, bias=False)),
                ("relu1", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(self.hidden_size // 4, output_dim_narr, bias=False)),
                ("relu2", nn.ReLU(inplace=True))
            ]))

        ##### end ####  CLIP ADAPTER  ############
        #############################################


        if self.temporal not in ['max', 'avg']:
            self.temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)
            nn.init.normal_(self.temporalEmbedding.weight, std=0.01)

        # temporal part is borrowed from github.com/ju-chen/Efficient-Prompt
        if self.temporal in ['attention', 'attention_with_full_frames', 'attention_with_full_frames_avg']:
            self.temporalModelling = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )
        elif self.temporal in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
            # within the group
            self.temporalModelling_intra_frame = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )
            # between the groups
            self.temporalModelling_inter_frame = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS_TEMP,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )
        elif self.temporal in ['multigroup_attention_mean']:
            self.temporalModelling_intra_frame = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT,
                bottle_neck=cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK,
                )

        self.define_temporal_backbone_part(cfg)

    def define_temporal_backbone_part(self, cfg):
        if self.temporal_backbone not in ['max', 'avg']:
            self.temporalEmbedding_backbone = torch.nn.Embedding(self.numF, self.hidden_size)
            nn.init.normal_(self.temporalEmbedding_backbone.weight, std=0.01)

        # temporal part is borrowed from github.com/ju-chen/Efficient-Prompt
        if self.temporal_backbone in ['attention', 'attention_with_full_frames', 'attention_with_full_frames_avg']:
            self.temporalModelling_backbone = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )
        elif self.temporal_backbone in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
            # within the group
            self.temporalModelling_intra_frame_backbone = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )
            # between the groups
            self.temporalModelling_inter_frame_backbone = TemporalModelling(
                width=self.hidden_size,
                layers=cfg.TRAINER.TEMPORAL.TFM_LAYERS,
                heads=cfg.TRAINER.TEMPORAL.TFM_HEADS,
                dropout=cfg.TRAINER.TEMPORAL.DROPOUT
                )


    def temporal_forward(self, visual_features, b, t):
        if self.segments:
            if self.temporal == 'attention':  # temporal modelling
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = visual_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal == 'attention_with_full_frames':
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = visual_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1).view(b, num_extra_frames*t, -1)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal == 'attention_with_full_frames_avg':
                num_extra_frames = 2
                t = t // num_extra_frames
                image_features = visual_features.view(b, t, num_extra_frames, -1)
                image_features = image_features.mean(dim=2)
                tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = image_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal in ['multigroup_attention_avg', 'multigroup_attention_reshape', 'multigroup_attention_mean']:
                num_extra_frames = 2
                t = t // num_extra_frames
                image_features = visual_features.view(b, t, num_extra_frames, -1)  # b x extra x t x c
                image_features = image_features.view(b * t, num_extra_frames, -1)
                image_features = image_features.transpose(0,1) # extra x (b * t) x c
                image_features = self.temporalModelling_intra_frame(image_features)
                if 'avg' in self.temporal:
                    image_features = image_features.mean(dim=0)
                    image_features = image_features.view(b, t, -1) # (b x t) x c -> b x t x c
                    tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)),'t c -> b t c', b=b)
                    image_features = image_features.view(b, t, -1)
                    image_features = image_features + tempEmbedding.to(self.device)
                    image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                    image_features = self.temporalModelling_inter_frame(image_features)
                    image_features = image_features.mean(dim=0)
                elif 'mean' in self.temporal:
                    image_features = image_features.mean(dim=0)
                    image_features = image_features.view(b, t, -1).mean(1) # (b x t) x c -> b x c
                elif 'reshape' in self.temporal:
                    image_features = image_features.view(num_extra_frames, b, t, -1) # extra x (b x t) x c -> extra x b x t x c
                    image_features = image_features.transpose(0,1) # extra x b x t x c  ->  b x extra x t x c
                    image_features = image_features.reshape(b, num_extra_frames * t, -1) # b x extra x t x c   -> b x (extra x t) x c
                    image_features = image_features.transpose(0,1) # b x (extra x t) x c  -> (extra x t) x b x c
                    image_features = self.temporalModelling_inter_frame(image_features)
                    image_features = image_features.mean(dim=0)
            elif self.temporal == 'avg':  # temporal modelling
                image_features = visual_features.view(b, t, -1).mean(dim=1)
            elif self.temporal == 'max':  # temporal modelling
                image_features = visual_features.view(b, t, -1).max(dim=1)[0]
            else:
                raise NotImplementedError('Check temporal function')
            if self.meta_net is not None:
                image_features = self.meta_net(image_features)
            image_features = image_features.unsqueeze(1)
        return image_features

    def temporal_backbone_forward(self, visual_features, b, t):
        if self.segments:
            if self.temporal_backbone == 'attention':  # temporal modelling
                tempEmbedding = einops.repeat(self.temporalEmbedding_backbone(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b)
                image_features = visual_features.view(b, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling_backbone(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal_backbone == 'attention_with_full_frames':
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding_backbone(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = visual_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1).view(b, num_extra_frames*t, -1)
                image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                image_features = self.temporalModelling_backbone(image_features)
                image_features = image_features.mean(dim=0)
            elif self.temporal_backbone in ['multigroup_attention_avg', 'multigroup_attention_reshape']:
                num_extra_frames = 2
                t = t // num_extra_frames
                tempEmbedding = einops.repeat(self.temporalEmbedding_backbone(torch.arange(self.numF).to(self.device)), 't c -> b t c', b=b*num_extra_frames)
                image_features = visual_features.view(b*num_extra_frames, t, -1)
                image_features = image_features + tempEmbedding.to(self.device)
                image_features = image_features.view(b, num_extra_frames, t, -1)  # b x extra x t x c
                image_features = image_features.transpose(0,1) # extra x b x t x c
                image_features = image_features.reshape(num_extra_frames, b * t, -1) # extra x (b x t) x c
                image_features = self.temporalModelling_intra_frame_backbone(image_features)
                if 'avg' in self.temporal_backbone:
                    image_features = image_features.mean(dim=0)
                    image_features = image_features.view(b, t, -1) # (b x t) x c -> b x t x c
                    image_features = image_features.transpose(0,1)  # b x t x c -> t x b x c
                    image_features = self.temporalModelling_inter_frame_backbone(image_features)
                    image_features = image_features.mean(dim=0)
                elif 'reshape' in self.temporal_backbone:
                    image_features = image_features.view(num_extra_frames, b, t, -1) # extra x (b x t) x c -> extra x b x t x c
                    image_features = image_features.transpose(0,1) # extra x b x t x c  ->  b x extra x t x c
                    image_features = image_features.reshape(b, num_extra_frames * t, -1) # b x extra x t x c   -> b x (extra x t) x c
                    image_features = image_features.transpose(0,1) # b x (extra x t) x c  -> (extra x t) x b x c
                    image_features = self.temporalModelling_inter_frame_backbone(image_features)
                    image_features = image_features.mean(dim=0)
            elif self.temporal_backbone == 'avg':  # temporal modelling
                image_features = visual_features.view(b, t, -1).mean(dim=1)
            elif self.temporal_backbone == 'max':  # temporal modelling
                image_features = visual_features.view(b, t, -1).max(dim=1)[0]
            else:
                raise NotImplementedError('Check temporal function')
        return image_features

    def class_sets_forward(self, class_sets, device=None, root=''):
        self.text_encoder.eval()
        if device is None:
            print('class sets', class_sets.keys())
        # create fixed classifiers that are modified by conditioning on the visual embeddings
        class_embeddings = {}
        for k,v in class_sets.items():
            # "#C C" into "Person"
            norm_txt = 'norm' if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM else 'nonorm'
            bakebone = self.cfg.MODEL.BACKBONE.NAME
            bakebone = bakebone.replace('/', '_')
            if self.prompt_prefix:
                prompt_path = self.prompt_prefix
                prompt_path = prompt_path.replace(' ', '_')
                path = os.path.join(root, f'{self.cfg.MODEL.BACKBONE.FRAMEWORK}_{bakebone}_{norm_txt}_{k}_{prompt_path}.pth')
            else:
                path = os.path.join(root, f'{self.cfg.MODEL.BACKBONE.FRAMEWORK}_{bakebone}_{norm_txt}_{k}.pth')
            if root and os.path.exists(path):
                class_embeddings[k] = torch.load(path).to(self.device)
            else:
                v = [p.replace("_", " ").strip() for p in v]
                if self.prompt_prefix:
                    v = [self.prompt_prefix + " " + name for name in v]
                # v = [" ".join(p.split()) for p in v]
                # v = [p.replace("#C C", "Person") for p in v]
                tokenized_text = torch.cat([clip.tokenize(p) for p in v])
                # tokenized_text = torch.cat([clip.tokenize(p.replace("_", " ")) for p in v])
                if device is not None:
                    tokenized_text = tokenized_text.to(device)
                with torch.no_grad():
                    embedding = self.token_embedding(tokenized_text).type(self.prompt_learner.dtype)
                    text_features = self.text_encoder(embedding, tokenized_text)
                    if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM:
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    class_embeddings_tmp = text_features.detach().unsqueeze(0)
                    class_embeddings_tmp = class_embeddings_tmp.to(self.device)
                    class_embeddings[k] = class_embeddings_tmp

                    torch.save(class_embeddings_tmp, path)

        return class_embeddings

    def text_forward(self, text):
        # if self.prompt_learner.ctx is None:
        v = [p.replace("_", " ") for p in text]
        if self.prompt_prefix:
            v = [self.prompt_prefix + " " + name for name in v]
        tokenized_text = torch.cat([clip.tokenize(p) for p in v]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_text).type(self.prompt_learner.dtype)
            text_features = self.text_encoder(embedding, tokenized_text)
        if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if self.cfg.TRAINER.DECOMP_COCOOP.MLP_NARRATIONS2:
            # mapped_narrations: BS(narr) x 1 x 512
            with torch.no_grad():
                mapped_narrations = self.meta_net_narrations(text_features.squeeze())
            if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM:
                mapped_narrations = mapped_narrations / mapped_narrations.norm(dim=-1, keepdim=True)
            mapped_narrations = mapped_narrations.unsqueeze(1)
            output = {'mapped_narrations': mapped_narrations.detach().cpu(), 'text_feat': text_features.detach().cpu()}
            return output

        return text_features.detach().cpu()

    def forward(self, image, test=False, narration=None, **kwargs):
        return_features = kwargs.get('return_features', False)
        if self.use_dino_features:
            ctx_image_features = image['dino']
            if self.cfg.TRAINER.DECOMP_COCOOP.VIS_CTX_INIT_NORM:
                ctx_image_features = ctx_image_features / ctx_image_features.norm(dim=-1, keepdim=True)
            image = image['not_dino']
        else:
            ctx_image_features = image
        b, t = 0,0
        if self.segments:
            if self.use_extracted_features:
                if self.cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
                    assert len(image.shape) == 3
                    b, t, c = image.shape
                    image = image.view(-1, c)
                elif self.cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
                    assert len(image.shape) == 2
                    b, c = image.shape
                    # image = image.view(-1, c)

            else:
                assert len(image.shape) == 5
                b, t, c, h, w = image.shape
                if self.cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
                    image = image.reshape(-1, c, h, w)
                elif self.cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
                    image = image.permute(0, 2, 1, 3, 4)

        if self.use_extracted_features:
            image_features = image.type(self.dtype)
        else:
            image_features = self.image_encoder(image.type(self.dtype))
            if self.cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
                image_features = image_features @ self.image_projection

        # breakpoint()
        b_dino, t_dino, c_dino = ctx_image_features.shape
        ctx_image_features = self.temporal_forward(ctx_image_features, b=b_dino, t=t_dino)
        ctx_vid2vid_features = None
        if isinstance(ctx_image_features, dict):
            ctx_vid2vid_features = ctx_image_features['image_features_vid2vid']
            ctx_image_features = ctx_image_features['image_features']
        elif self.cfg.TRAINER.DECOMP_COCOOP.VID2VID:
            ctx_vid2vid_features = ctx_image_features.squeeze(1)

        if self.cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
            image_features = self.temporal_backbone_forward(image_features, b=b, t=t)
        if self.cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
            pass

        if self.cfg.TRAINER.DECOMP_COCOOP.SKIP_TEXT_ENCODER:
            text_visual_features = ctx_image_features
        else:
            visual_prompts = self.prompt_learner(ctx_vis_features=ctx_image_features)
            text_visual_features = self.text_encoder(visual_prompts, self.tokenized_visual_prompts).unsqueeze(1)

        if ctx_vid2vid_features is not None:
            if self.cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER:
                ratio = 0.2
                image_features = ratio * image_features + (1-ratio) * ctx_vid2vid_features
            else:
                if self.cfg.TRAINER.DECOMP_COCOOP.VID2VID_NORM:
                    ctx_vid2vid_features = ctx_vid2vid_features / ctx_vid2vid_features.norm(dim=-1, keepdim=True)
                image_features = self.clip_adapter_ratio * image_features + (1 - self.clip_adapter_ratio) * ctx_vid2vid_features
        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_NORM:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_CTX_NORM:
            text_visual_features = text_visual_features / text_visual_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if return_features:
            output_features = {}
            output_features = {'visual_ctx_feat': (self.prompt_learner.scale_factor_vis_ctx * text_visual_features).detach().cpu()}
            output_features.update({'image_feat': image_features.detach().cpu()})

        image_features = image_features.unsqueeze(1)

        output = {}
        if test and self.test_class_embeddings is not None:
            class_embeddings = self.test_class_embeddings
        else:
            class_embeddings = self.class_embeddings

        if 'test_egtea' in kwargs and self.test_class_embeddings_egtea is not None:
            class_embeddings = self.test_class_embeddings_egtea

        for k,text_features in class_embeddings.items():

            if self.cfg.TRAINER.DECOMP_COCOOP.MLP_LY_TXT2TXT:
                if self.llama_conditioning:
                    if test and self.test_class_embeddings is not None:
                        text_ctx = self.llama_embed_test[k]
                    else:
                        text_ctx = self.llama_embed[k]
                else:
                    text_ctx = text_features
                if self.cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER:
                    ratio = 0.2
                    # breakpoint()
                    text_features = ratio * text_features + (1-ratio) * self.meta_net_txt2txt(text_ctx)
                else:
                    text_ctx = self.meta_net_txt2txt(text_ctx)
                    if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_CTX_NORM:
                        text_ctx = text_ctx / text_ctx.norm(dim=-1, keepdim=True)
                    text_features = self.clip_adapter_ratio * text_features + (1-self.clip_adapter_ratio) * text_ctx
                if self.cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM2:
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_features = text_features + self.prompt_learner.scale_factor_vis_ctx * text_visual_features

            text_shape = text_features.shape
            text_features = text_features.view(-1, text_shape[-1])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(*text_shape)

            if not self.cfg.TRAINER.DECOMP_COCOOP.VISUAL_NORM:
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_features_norm = image_features


            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * ((image_features_norm * text_features).sum(-1))
            # logits = ((image_features.squeeze() @ text_features.squeeze().t()).sum(-1))
            output[k] = logits

        if return_features:
            return output, output_features

        return output


@TRAINER_REGISTRY.register()
class XMIC(TrainerXEpic):
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
        assert isinstance(self.num_classes, dict)
        print('NUM CLASSES1', self.num_classes, flush=True)
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
        self.use_dino_features = self.cfg.DATALOADER.USE_DINO_FEATURES

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

        if self.cfg.TEST.RETRIEVAL:
            self.text_val_dataloader = dm.text_val_dataloader


    def build_model(self, init_weights=None):
        cfg = self.cfg
        # classnames = self.dm.dataset.classnames
        class_sets = self.dm.dataset.classnames
        # print('BUILD MODEL', classnames, flush=True)
        # print('BUILD MODEL', len(classnames), flush=True)
        print('Output dir', cfg.OUTPUT_DIR)

        if cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
            print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
            clip_model = load_clip_to_cpu(cfg)

        elif cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
            print(f"Loading Lavila (backbone: {cfg.MODEL.BACKBONE.NAME})")
            ckpt_path = cfg.MODEL.BACKBONE.CKPT_PATH

            ckpt = torch.load(ckpt_path, map_location='cpu')

            # create model
            state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                state_dict[k.replace('module.', '')] = v

            old_args = ckpt['args']
            print('=> creating model: {}'.format(old_args.model))
            clip_model = getattr(models, old_args.model)(
                text_use_cls_token=old_args.use_cls_token,
                project_embed_dim=old_args.project_embed_dim,
                gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
                timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
                timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
                freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
                freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
                num_frames=cfg.DATALOADER.FRAMES_PER_SEGMENT,
                drop_path_rate=0,
            )
            # clip_model.to(self.device)

            if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
                # inflate weight
                print('=> inflating PE in models due to different frame numbers')
                state_dict = inflate_positional_embeds(
                    clip_model.state_dict(), state_dict,
                    num_frames=cfg.DATALOADER.FRAMES_PER_SEGMENT,
                    load_temporal_fix='bilinear',
                )

            clip_model.load_state_dict(state_dict, strict=True)
            print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(cfg.MODEL.BACKBONE.CKPT_PATH, ckpt['epoch'], ckpt['best_acc1']))

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if 'narration' in self.cfg.DATASET.LABEL_SUBTYPES:
            self.cliploss_weight = cfg.TRAINER.DECOMP_COCOOP.CLIPLOSS_W



        print("Building custom CLIP")

        if self.cross_eval_egtea:
            egtea_classes = self.cross_dm_egtea.dataset.classnames
        else:
            egtea_classes = None
        if self.cross_eval:
            test_class_sets = self.cross_dm.dataset.classnames
            self.model = CustomCLIP(cfg, class_sets, clip_model, test_class_sets=test_class_sets, egtea_classes=egtea_classes)
        else:
            self.model = CustomCLIP(cfg, class_sets, clip_model, egtea_classes=egtea_classes)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and 'temporal' not in name and 'meta_net' not in name and 'logit_scale' not in name:
                param.requires_grad_(False)
            else:
                print(f'Params with grad: {name}')

        if cfg.MODEL.INIT_WEIGHTS:
            if init_weights is not None:
                load_pretrained_weights(self.model, init_weights)
            else:
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
            self.model.tokenized_visual_prompts = self.model.tokenized_visual_prompts.cuda(local_rank)

            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
            print(f'Done DDP {local_rank}')
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model



    def model_inference(self, input, test=False, **kwargs):
        only_text = kwargs.get('only_text', False)
        if only_text:
            return self.model_without_ddp.text_forward(input)
        return self.model_without_ddp(input, test=test, **kwargs)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                loss = None
                output = self.model(image)

                for label_type in output.keys():
                    tmp_loss = F.cross_entropy(output[label_type], label[label_type])
                    if loss is None: loss = tmp_loss
                    else: loss += tmp_loss

            if self.cfg.TRAINER.ACCUMULATION_STEPS == 1:
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                if (self.batch_idx + 1) % self.cfg.TRAINER.ACCUMULATION_STEPS == 0:
                    self.optim.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.scaler.scale(loss).backward()

        else:
            loss = None
            output = self.model(image)
            for label_type in output.keys():
                tmp_loss = F.cross_entropy(output[label_type], label[label_type])
                if loss is None:
                    loss = tmp_loss
                else:
                    loss += tmp_loss
            # loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)


        loss_summary = {
            "loss": loss.item()
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
            input = input.to(self.device)
            if 'label' in batch:
                label = batch["label"]
                label = label.to(self.device)
            else:
                label = {}
                for label_type in ['noun', 'verb', 'action']:
                    label_tmp = batch[f'{label_type}_label']
                    label[label_type] = label_tmp.to(self.device)
                if 'narration' in self.cfg.DATASET.LABEL_SUBTYPES:
                    label['narration'] = batch['narration']

            if self.use_dino_features:
                dino = batch["dino"]
                dino = dino.to(self.device)
                output = {'not_dino': input,
                          'dino': dino}
                return output, label
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
            input = input.to(self.device)
            narration_id = batch['narration_id']

            if 'label' in batch:
                label = batch["label"]
                label = label.to(self.device)
            else:
                label = {}
                for label_type in ['noun', 'verb', 'action']:
                    label_tmp = batch[f'{label_type}_label']
                    label[label_type] = label_tmp.to(self.device)
                if 'narration' in self.cfg.DATASET.LABEL_SUBTYPES:
                    label['narration'] = batch['narration']

            if self.use_dino_features:
                dino = batch["dino"]
                dino = dino.to(self.device)
                output = {'not_dino': input,
                          'dino': dino}
                return output, label, narration_id

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

            # Ignore fixed token vectors prompt_learner.token_prefix
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]


            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            if "prompt_learner.test_token_prefix" in state_dict:
                del state_dict["prompt_learner.test_token_prefix"]

            if "prompt_learner.test_token_suffix" in state_dict:
                del state_dict["prompt_learner.test_token_suffix"]

            if "prompt_learner.token_embedding" is state_dict:
                del state_dict["prompt_learner.token_embedding"]

            if "token_embedding" is state_dict:
                del state_dict["token_embedding"]


            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
