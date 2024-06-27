import os.path as osp
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXEpic
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
# from dassl.data import DatasetSegmentWrapper, DataManager, Ego4DDataManager
from dassl.data import DatasetSegmentWrapper, DataManager
from dassl.evaluation import build_evaluator

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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, tokenized_prompts):
        prompts = self.token_embedding(tokenized_prompts).type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # NOTE: this line is not yet completely clear

        return x



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, test_classnames=None):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.segments = cfg.DATALOADER.SEGMENTS

        prompt_prefix = 'The photo of a'
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name for name in classnames] # NOTE: deleted "." in the end of sentence

        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        if test_classnames is not None:
            test_classnames = [name.replace("_", " ") for name in test_classnames]
            test_name_lens = [len(_tokenizer.encode(name)) for name in test_classnames]
            test_prompts = [prompt_prefix + " " + name for name in test_classnames] # NOTE: deleted "." in the end of sentence

            self.tokenized_prompts_test = torch.cat([clip.tokenize(p) for p in test_prompts])


    def forward(self, image, test=False):
        if self.segments:
            assert len(image.shape) == 5
            b, t, c, h, w = image.shape
            image = image.reshape(-1, c, h, w)

        image_features = self.image_encoder(image.type(self.dtype))

        if self.segments:
            image_features = image_features.view(b, t, -1).mean(dim=1)

        # print("Encoder", self.text_encoder.device, flush=True)
        # print('Prompts', self.tokenized_prompts.device, flush=True)
        if test:
            text_features = self.text_encoder(self.tokenized_prompts_test)
        else:
            text_features = self.text_encoder(self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class ClipFT(TrainerXEpic):
    """CLIP fine-tuning.

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

        print("Building custom CLIP")
        if cfg.DATASET.LABEL_TYPE == 'noun' and cfg.DATASET.SUBSET == 'seen_nouns':
            test_classnames = self.dm.dataset.test_classes
            self.model = CustomCLIP(cfg, classnames, clip_model, test_classnames=test_classnames)
        else:
            test_classnames = None
            self.model = CustomCLIP(cfg, classnames, clip_model)

        # print("Turning off gradients in both the image and the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "prompt_learner" not in name:
        #         param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.tokenized_prompts = self.model.tokenized_prompts.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("clip_ft", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        if cfg.DISTRIBUTED:
            print(f"Multiple GPUs detected (n_gpus={get_world_size()}), use all of them!")
            # print('Apply SyncBN', flush=True)
            # self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            local_rank = int(os.environ["LOCAL_RANK"])
            self.local_rank = local_rank
            # torch.cuda.set_device(local_rank)
            print(f'Apply DDP {local_rank}', flush=True)
            self.model.cuda(local_rank)
            self.model.tokenized_prompts = self.model.tokenized_prompts.cuda(local_rank)
            if test_classnames is not None:
                self.model.tokenized_prompts_test = self.model.tokenized_prompts_test.cuda(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)
            print(f'Done DDP {local_rank}')
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model


    def model_inference(self, input, test=False):
        return self.model_without_ddp(input, test=test)


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
                loss = F.cross_entropy(output, label, label_smoothing=self.cfg.OPTIM.LABEL_SMOOTHING)
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
            output = self.model(image)
            # print('OUTPUT', output, flush=True)
            # print('MAX output', output.max(dim=-1), flush=True)
            # print('output.shapes', output.shape, flush=True)
            # print('LABEL', label, flush=True)
            loss = F.cross_entropy(output, label, label_smoothing=self.cfg.OPTIM.LABEL_SMOOTHING)
            # print('LOSS', loss, flush=True)
            self.model_backward_and_update(loss)
            if self.cfg.TRAINER.ACCUMULATION_STEPS != 1:
                raise NotImplementedError('Accumulation steps only for amp precision')

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch):
        if 'img' in batch:
            input = batch["img"]
            label = batch["label"]
            input = input.to(self.device)
            label = label.to(self.device)
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
                full_path = osp.expanduser(model_path)
                print(f'MODEL FULL PATH {full_path}', flush=True)
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
