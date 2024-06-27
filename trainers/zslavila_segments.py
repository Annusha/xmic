import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DatasetSegmentWrapper, DataManager
from lavila import models
from lavila.tokenizer import SimpleTokenizer
from lavila.utils import inflate_positional_embeds
from collections import OrderedDict

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

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
}


# Namespace(batch_size=32, betas=[0.9, 0.999], caption_loss_scale=1.0, clip_grad_type='norm', clip_grad_value=None, clip_length=4,
#  clip_stride=16, comment='', contrastive_loss_scale=1.0, contrastive_use_vissl=True, dataset='ego4d', dataset_aux='ego4d',
# disable_amp=False, dist_backend='nccl', dist_url='file:///checkpoint/yuezh/experiments/lavid_ssl/1fc612d0639d44e3baae2e51ac846341_init', distributed=True,
#  epochs=5, eps=1e-08, eval_freq=99, eval_in_middle=2000, evaluate=False, exclude_nodes_list=[], find_unused_parameters=False, fix_lr=True,
# freeze_pseudo_scale=True, freeze_temperature=True, gpu=0, gt_percentage=0.5, job_dir=PosixPath('/checkpoint/yuezh/experiments/lavid_ssl/%j'),
# load_visual_pretrained=None, local_rank=0, lr=3e-05, lr_end=1e-05, lr_start=1e-06, metadata='/checkpoint/yuezh/LaVid/metadata/paraphrase/ego4d_train_v5.paraphrase.no_punkt_top3.pkl',
# metadata_aux=['/private/home/yuezh/LaVid/metadata/ego4d_train_v5.narrator_1+2.multinomial.nucleus_0.95.temperature_0.7.return_10.pseudo_labeled_by_flamingo_63690737_ckpt_0001.pkl'],
# model='CLIP_OPENAI_TIMESFORMER_BASE', narration_selection='random', ngpus=8, nodes=4, norm_embed=True, num_hard_neg=0,
#  output_dir=PosixPath('/checkpoint/yuezh/experiments/lavid_ssl/65610026'), partition='learnlab', print_freq=10, project_embed_dim=256,
#  pseudo_scale_init=0.07, rank=0, re_prob=0.0, resume='', root='/checkpoint/kalyanv/ego4d/2022-02-01/', root_aux='/checkpoint/kalyanv/ego4d/2022-02-01/',
# save_freq=1, seed=0, sparse_sample=False, start_epoch=4, temperature_init=0.07, timeout=4200, untie_bert_encoder=False, untie_word_embedding=False,
# untie_word_embedding_and_lmhead=False, update_freq=1, use_checkpoint=False, use_cls_token=False, use_rand_aug=False, use_volta32=True, use_zero=False,
#  wandb=False, warmup_epochs=1, wd=0.01, workers=10, world_size=32)

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

@TRAINER_REGISTRY.register()
class ZeroshotLavilaSegments(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        lab2cname = self.dm.dataset.lab2cname
        assert len(lab2cname) == len(classnames)
        assert max(lab2cname) == len(classnames)-1

        ckpt_path = cfg.MODEL.BACKBONE.CKPT_PATH

        ckpt = torch.load(ckpt_path, map_location='cpu')

        # create model
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        print('=> creating model: {}'.format(old_args.model))
        model = getattr(models, old_args.model)(
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
        model.to(self.device)

        if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
            # inflate weight
            print('=> inflating PE in models due to different frame numbers')
            state_dict = inflate_positional_embeds(
                model.state_dict(), state_dict,
                num_frames=cfg.DATALOADER.FRAMES_PER_SEGMENT,
                load_temporal_fix='bilinear',
            )

        model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(cfg.MODEL.BACKBONE.CKPT_PATH, ckpt['epoch'], ckpt['best_acc1']))

        tokenizer = SimpleTokenizer()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(lab2cname[lab].replace("_", " ")) for lab in sorted(lab2cname.keys())]
        print(f"Prompts: {prompts}")

        prompts = torch.cat([tokenizer(p).view(-1, 77) for p in prompts])
        # prompts = tokenizer(prompts)
        prompts = prompts.to(self.device)
        # prompts = prompts.view(-1, 77).contiguous()

        with torch.no_grad():
            text_features = get_model(model).encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        print('Prompts shape', text_features.shape, flush=True)

        self.text_features = text_features
        self.clip_model = model
        self.segments = cfg.DATALOADER.SEGMENTS

    def build_data_loader(self):
        dm = DataManager(self.cfg, dataset_wrapper=DatasetSegmentWrapper)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        print('NUM CLASSES', self.num_classes, flush=True)
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def model_inference(self, image):
        assert len(image.shape) == 5
        b, t, c, h, w = image.shape
        image = image.permute(0,2,1,3,4)

        image_features = get_model(self.clip_model).encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # image_features = image_features.view(b, t, -1)
        # if self.cfg.DATALOADER.FEATURE_EXTRACT:
        #     return image_features
        # image_features = image_features.mean(dim=1)
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # logit_scale = self.clip_model.logit_scale.exp()
        logits = image_features @ self.text_features.t()
        return logits

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        # print("INPUT SHAPE", input.shape, flush=True)
        # print('Labels', label, flush=True)

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    # def feature_extract(self, image):
    #     image_features = self.clip_model.encode_image(image)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     return image_features
