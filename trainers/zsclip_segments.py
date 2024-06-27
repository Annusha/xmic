import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXEpic
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DatasetSegmentWrapper, DataManager

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
    # "Ego4DRecognitionWrapper": "a photo of a {}"
    "Ego4DRecognitionWrapper": "a photo of a {} action"
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIPSegments(TrainerXEpic):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        lab2cname = self.dm.dataset.lab2cname
        assert len(lab2cname) == len(classnames)
        assert max(lab2cname) == len(classnames)-1

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)




        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(lab2cname[lab].replace("_", " ")) for lab in sorted(lab2cname.keys())]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        print('Prompts shape', text_features.shape, flush=True)

        self.text_features = text_features
        self.test_text_features = None
        self.clip_model = clip_model
        self.segments = cfg.DATALOADER.SEGMENTS
        self.dtype = clip_model.dtype
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES

        if (cfg.DATASET.LABEL_TYPE == 'noun' and cfg.DATASET.SUBSET == 'seen_nouns') or \
                (cfg.DATASET.LABEL_TYPE == 'action' and cfg.DATASET.SUBSET == 'ek-55'):
            print("Building test CLIP prompts")
            test_lab2cname = self.dm.dataset.test_lab2cname
            test_prompts = [temp.format(test_lab2cname[lab].replace("_", " ")) for lab in sorted(test_lab2cname.keys())]
            print(f"TEST Prompts: {prompts}")
            test_prompts = torch.cat([clip.tokenize(p) for p in test_prompts])
            test_prompts = test_prompts.to(self.device)

            with torch.no_grad():
                test_text_features = clip_model.encode_text(test_prompts)
                test_text_features = test_text_features / test_text_features.norm(dim=-1, keepdim=True)

            print('TEST Prompts shape', test_text_features.shape, flush=True)
            self.test_text_features = test_text_features



    def build_data_loader(self):
        if 'Ego4D' in self.cfg.DATASET.NAME:
            from dassl.data import Ego4DDataManager
            dm = Ego4DDataManager(self.cfg)
        else:
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

    def model_inference(self, image, test=None):

        if self.use_extracted_features:
            assert len(image.shape) == 3
            b, t, c = image.shape
            image = image.view(-1, c)
            image_features = image.type(self.dtype)
        else:
            assert len(image.shape) == 5
            b, t, c, h, w = image.shape
            image = image.reshape(-1, c, h, w)
            image_features = self.clip_model.encode_image(image)

        # assert len(image.shape) == 5
        # b, t, c, h, w = image.shape
        # image = image.reshape(-1, c, h, w)

        # image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(b, t, -1)
        # if self.cfg.DATALOADER.FEATURE_EXTRACT:
        #     return image_features
        image_features = image_features.mean(dim=1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        if test and self.test_text_features is not None:
            logits = logit_scale * image_features @ self.test_text_features.t()
        else:
            logits = logit_scale * image_features @ self.text_features.t()
        return logits

    # def parse_batch_test(self, batch):
    #     input = batch["img"]
    #     label = batch["label"]
    #
    #     # print("INPUT SHAPE", input.shape, flush=True)
    #     # print('Labels', label, flush=True)
    #
    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #
    #     return input, label

    def parse_batch_test(self, batch):
        if 'img' in batch:
            input = batch["img"]
            label = batch["label"]
            narration_id = batch['narration_id']

            input = input.to(self.device)
            label = label.to(self.device)

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

    # def feature_extract(self, image):
    #     image_features = self.clip_model.encode_image(image)
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     return image_features
