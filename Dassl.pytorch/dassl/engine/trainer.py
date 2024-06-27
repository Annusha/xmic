import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import neptune
from collections import defaultdict

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from dassl.utils.dist_utils import get_world_size, get_rank


class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            print('build classifier', num_classes)
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        print('Before classifier: ', f.shape)
        y = self.classifier(f)
        print('After classifier: ', y.shape)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None
        self._neptune = None
        # self.clip_grad = 0.1

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name="", iter=0
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result,
                    "iter": iter,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name],
                local_rank=self.local_rank
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                print(f'MODEL FULL PATH {osp.expanduser(model_path)}', flush=True)
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def init_neptune(self, project, api_token, with_id, mode, tags):
        self._neptune = neptune.init_run(
            project=project,
            api_token=api_token,
            with_id=with_id,
            mode=mode,
            tags=tags,
        )

    def neptune_log_cfg(self, cfg):
        if self._neptune is not None:
            def _r(_cfg, name=''):
                if isinstance(_cfg, dict):
                    for k,v in _cfg.items():
                        _r(v, name=f"{name}/{k}")
                else:
                    self._neptune[name] = _cfg
            _r(cfg)

    def neptune_log_args(self, args):
        if self._neptune is not None:
            optkeys = list(args.__dict__.keys())
            optkeys.sort()
            for key in optkeys:
                self._neptune[key] = args.__dict__[key]

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

        if self._neptune is not None:
            self._neptune[tag].append(value=scalar_value, step=global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def test_cross(self, split='val', accuracy=0):
        pass

    def test_cross_egtea(self, split='val', accuracy=0):
        pass

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        # if self.clip_grad > 0:
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        #     print('GRAD CLIP', flush=True)
        self.model_update(names)





class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.local_rank = 0
        self.cross_eval = False
        self.ego_mcq_eval = False
        self.cross_eval_egtea = False
        self.cross_dm = None

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        print('Build Eval', flush=True)
        self.build_evaluator_trainer()
        self.best_result = -np.inf

    # SimpleTrainer
    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    # SimpleTrainer
    def build_evaluator_trainer(self):
        self.evaluator = build_evaluator(self.cfg, lab2cname=self.lab2cname)

    # SimpleTrainer
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.text_val_dataloader = None
        self.cross_text_val_dataloader = None

        self.num_classes = dm.num_classes
        print('TRAINER NUM CLASSES', self.num_classes, flush=True)
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    # SimpleTrainer
    def build_model(self, args=None):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            # print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            # Apply SyncBN
            print('Apply SyncBN', flush=True)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # device = torch.device(args.device + f':{args.gpu}')

            # not exactly correct!!! valid only for one node!
            local_rank = int(os.environ["LOCAL_RANK"])
            self.local_rank = local_rank

            # torch.cuda.set_device(local_rank)
            # model = model.cuda(args.gpu)
            # loss = loss.cuda(args.gpu)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
            self.model_without_ddp = self.model.module
            # self.model = nn.DataParallel(self.model)

    # SimpleTrainer
    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    # SimpleTrainer
    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        if self.local_rank == 0:
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()
        if self.cfg.TRAIN.TEST_BEFORE_TRAIN:
            accuracy = self.test(split="val")
            # accuracy = 0
            if self.cross_eval:
                self.test_cross(split='val', accuracy=accuracy)
            if self.cross_eval_egtea:
                self.test_cross_egtea(split='val', accuracy=accuracy)
                self.test_cross_egtea(split='test', accuracy=accuracy)

    # SimpleTrainer
    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            accuracy = self.test(split="val")
            if self.cross_eval:
                self.test_cross(split='val', accuracy=accuracy)
            if self.cross_eval_egtea:
                self.test_cross_egtea(split='val', accuracy=accuracy)
                self.test_cross_egtea(split='test', accuracy=accuracy)
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    # SimpleTrainer
    def after_epoch(self):
        print('after epoch', self.epoch)
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        save_every_last_epoch = self.cfg.TRAIN.CHECKPOINT_EVERY_LAST_EPOCH
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if save_every_last_epoch:
            self.save_model(self.epoch, self.output_dir, model_name="model-cur-last.pth.tar")

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split=self.cfg.TEST.VAL_SPLIT)
            if self.cross_eval:
                curr_result = self.test_cross(split='val', accuracy=curr_result)
            if self.cross_eval_egtea:
                self.test_cross_egtea(split='val')
                self.test_cross_egtea(split='test')
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
            if self.cfg.TEST.EVERY_EPOCH:
                accuracy = self.test(split=self.cfg.TEST.TEST_SPLIT)
                if self.cross_eval:
                    curr_result = self.test_cross(split='val', accuracy=accuracy)
                if self.cross_eval_egtea:
                    self.test_cross_egtea(split='val')
                    self.test_cross_egtea(split='test')

        save_early_epochs = (self.epoch + 1) <= self.cfg.TRAIN.CHECKPOINT_EARLY_EPOCHS
        if meet_checkpoint_freq or last_epoch or save_early_epochs:
            self.save_model(self.epoch, self.output_dir)


    # SimpleTrainer
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        self.evaluator.synchronize_between_processes()
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        print(f'cfg NAME {self.cfg.NAME}')
        # return list(results.values())[0]
        return results['accuracy']

    # SimpleTrainer
    def model_inference(self, input):
        return self.model(input)

    # SimpleTrainer
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    # SimpleTrainer
    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        print('Start epoch',self.num_batches, flush=True)
        end = time.time()
        # print(f'R{get_rank()} Before first batch', len(self.train_loader_x))
        # print(f'R{get_rank()} Before first batch', self.train_loader_x)
        # print(f'R{get_rank()} Before first batch', self.train_loader_x.dataset.__getitem__(0))
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            # print('After first batch')
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            if meet_freq or only_few_batches:
                for name, meter in losses.meters.items():
                    self.write_scalar("train/" + name, meter.avg, n_iter)
                self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    # TrainerX
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain


class TrainerXEpic(SimpleTrainer):
    """A base trainer using labeled data only."""
    # TrainerXEpic
    def build_evaluator_trainer(self):
        train_counts = None
        dist_splits_novel_base = None
        dist_splits = None
        n_test_classes = None
        n_crosseval_classes = None
        dist_crosseval_splits_novel_base = None
        if self.cfg.TEST.BASE_NOVEL_EVAL:
            n_test_classes = self.dm.dataset.n_test_classes
            dist_splits_novel_base = self.dm.dataset.dist_splits_novel_base
        if self.cfg.TEST.LT_EVAL:
            train_counts = self.dm.dataset.train_class_counts
            dist_splits = self.dm.dataset.dist_splits
        if self.cfg.TEST.CROSS_DATASET.EVAL:
            n_crosseval_classes = self.cross_dm.dataset.n_test_classes
            dist_crosseval_splits_novel_base = self.cross_dm.dataset.dist_splits_novel_base
        if self.cfg.TEST.TSNE.SAVE:
            self.narration_id_collection = self.dm.dataset.narration_id_collection
            self.label_collection = self.dm.dataset.label_collection

        # if self.cfg.TEST.LT_EVAL:
        #     print('Build evaluator with Long-Tailed evauation')
        self.evaluator = build_evaluator(
            self.cfg,
            lab2cname=self.lab2cname,
            train_counts=train_counts,
            dist_splits=dist_splits,
            dist_splits_novel_base=dist_splits_novel_base,
            n_test_classes=n_test_classes,
            n_crosseval_classes=n_crosseval_classes,
            dist_crosseval_splits_novel_base=dist_crosseval_splits_novel_base,
        )

        # else:
        #     self.evaluator = build_evaluator(self.cfg, lab2cname=self.lab2cname)

    def during_epoch(self, iter):
        print('save and eval during epoch', self.epoch)
        print('iter', iter)
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        save_every_last_epoch = self.cfg.TRAIN.CHECKPOINT_EVERY_LAST_EPOCH
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if save_every_last_epoch:
            self.save_model(self.epoch, self.output_dir, model_name="model-cur-last.pth.tar")

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split=self.cfg.TEST.VAL_SPLIT)
            if self.cross_eval:
                curr_result = self.test_cross(split='val', accuracy=curr_result)
            if self.cross_eval_egtea:
                self.test_cross_egtea(split='val')
                self.test_cross_egtea(split='test')
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
            if self.cfg.TEST.EVERY_EPOCH:
                accuracy = self.test(split=self.cfg.TEST.TEST_SPLIT)
                if self.cross_eval:
                    curr_result = self.test_cross(split='val', accuracy=accuracy)
                if self.cross_eval_egtea:
                    self.test_cross_egtea(split='val')
                    self.test_cross_egtea(split='test')


    # TrainerXEpic
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        print('Start epoch', self.num_batches, flush=True)

        end = time.time()
        # print(f'R{get_rank()} Before first batch', len(self.train_loader_x))
        # print(f'R{get_rank()} Before first batch', self.train_loader_x)
        # print(f'R{get_rank()} Before first batch', self.train_loader_x.dataset.__getitem__(0))
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            # print('After first batch')
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            meet_freq_eval = (self.batch_idx + 1) % self.cfg.TRAIN.EVAL_FREQ == 0
            if meet_freq_eval:
                self.during_epoch(self.batch_idx)

            n_iter = self.epoch * self.num_batches + self.batch_idx
            if meet_freq or only_few_batches:
                for name, meter in losses.meters.items():
                    self.write_scalar("train/" + name, meter.avg, n_iter)
                self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    # TrainerXEpic
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

    def tsne(self, feat_collection, labels_collection):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                     'tab:gray', 'tab:olive', 'tab:cyan']
        markers_map = ["." , "," , "o" , "v" , "^" , "<", ">"]

        frames_per_seg = feat_collection[0].shape[0]
        feat_collection = np.concatenate(feat_collection)

        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(feat_collection)

        labels, markers = self.label_collection

        # X_embedded = X_embedded.reshape(len(labels), frames_per_seg, -1)
        fig, ax = plt.subplots()
        markers2idx = {}
        labels2idx = {}
        for idx, label_marker in enumerate(zip(labels, markers)):
            label = label_marker[0]
            marker = label_marker[1]
            x, y = X_embedded[idx*frames_per_seg: (idx+1)*frames_per_seg,0], X_embedded[idx*frames_per_seg: (idx+1)*frames_per_seg,1]
            if label not in labels2idx:
                labels2idx[label] = len(labels2idx)
            if marker not in markers2idx:
                markers2idx[marker] = len(markers2idx)
            label_idx = labels2idx[label]
            marker_idx = markers2idx[marker]
            ax.scatter(x, y, c=color_map[label_idx], marker=markers_map[marker_idx], label=f'{label}_{marker}', alpha=0.5)

        # ax.legend()
        # ax.grid(True)

        save_path = '/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/tsne/'
        plt.savefig(save_path + f"{self.cfg.TEST.TSNE.NAME}.pdf", dpi=100)


    # TrainerXEpic
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # feat_collection = defaultdict(list)
        # narration_ids_collection = []
        label_types = None
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, narration_id = self.parse_batch_test(batch)
            if self.cfg.TEST.RETRIEVAL:
                output, features = self.model_inference(input, test=split=='test', return_features=True)
                self.evaluator.process_retrieval(narration_id, features)

                ###############################################################
                ###################        TSNE         #######################
                ###############################################################
                # it was for tsne collection
                # for narr_idx, narr_id in enumerate(narration_id):
                    # if narr_id in self.narration_id_collection: # it was for tsne collection
                    #     feat_collection.append(features[narr_idx])
                    #     labels_collection.append(self.label_collection[narr_idx])
                ###############################################################

            else:
                output = self.model_inference(input, test=split=='test')
            if isinstance(output, dict):
                label_types = list(output.keys()) if label_types is None else label_types
                for k in output.keys():
                    self.evaluator.process(output[k], label[k], label_type=k)
                    self.evaluator.process_epic(output[k], narration_id, label_type=k)
            else:
                self.evaluator.process(output, label)
                self.evaluator.process_epic(output, narration_id)
                # if batch_idx == 10:
                #     break

        # if self.cfg.TEST.TSNE.SAVE:
        #     self.tsne(feat_collection, labels_collection)

        if self.cfg.TEST.RETRIEVAL:
            # evaluate separately only sentences without images
            text_data_loader = self.text_val_dataloader
            for batch_idx, batch in enumerate(tqdm(text_data_loader)):
                text_features = self.model_inference(batch, only_text=True)
                self.evaluator.process_retrieval_text(text_features)



        self.evaluator.synchronize_between_processes()
        self.evaluator.save_epic(self.local_rank, self.output_dir)

        if self.cfg.DATALOADER.EGOCLIP.EVAL:
            pass

        if label_types is None:
            results = self.evaluator.evaluate(split)

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)

            # retrieval
            if self.cfg.TEST.RETRIEVAL:
                print('Evaluate retrieval')
                self.evaluator.evaluate_retrieval_epic()

            print(f'cfg NAME {self.cfg.NAME}')
        else:
            for label_type in label_types:
                results = self.evaluator.evaluate(split, label_type=label_type)

                for k, v in results.items():
                    tag = f"{split}/{k}/{label_type}"
                    self.write_scalar(tag, v, self.epoch)


                print(f'cfg NAME {self.cfg.NAME}')
                # return list(results.values())[0]
            # retrieval
            if self.cfg.TEST.RETRIEVAL:
                print('Evaluate retrieval')
                results_ret = self.evaluator.evaluate_retrieval_epic()

                for k, v in results_ret.items():
                    tag = f"{split}/{k}/{label_type}"
                    self.write_scalar(tag, v, self.epoch)
                
            return list(results.values())[0]

    # TrainerXEpic
    @torch.no_grad()
    def test_cross(self, split='test', accuracy=0):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.CROSS_DATASET.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.cross_val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.cross_test_loader

        print(f"CROSS DATASET Evaluate on the *{split}* set")
        label_types = None
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, narration_id = self.parse_batch_test(batch)
            if self.cfg.TEST.CROSS_DATASET.RETRIEVAL:
                output, features = self.model_inference(input, test=True, return_features=True)
                self.evaluator.process_retrieval(narration_id, features)
            else:
                output = self.model_inference(input, test=True)

            if isinstance(output, dict):
                label_types = list(output.keys()) if label_types is None else label_types
                for k in output.keys():
                    self.evaluator.process(output[k], label[k], label_type=k)
                    self.evaluator.process_epic(output[k], narration_id, label_type=k)
            else:
                self.evaluator.process(output, label)
                self.evaluator.process_epic(output, narration_id)
            # if batch_idx == 10:
            #     break

        if self.cfg.TEST.CROSS_DATASET.RETRIEVAL:
            # evaluate separately only sentences without images
            text_data_loader = self.cross_text_val_dataloader
            for batch_idx, batch in enumerate(tqdm(text_data_loader)):
                text_features = self.model_inference(batch, only_text=True)
                self.evaluator.process_retrieval_text(text_features)

        self.evaluator.synchronize_between_processes()
        self.evaluator.save_epic(self.local_rank, self.output_dir)

        if label_types is None:
            results = self.evaluator.evaluate(split, prefix='cross/', accuracy_old=accuracy)
    
            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)
    
            print(f'CROSS DATASET cfg NAME {self.cfg.NAME}')
            return results['hm_acc']
        else:
            for label_type in label_types:
                results = self.evaluator.evaluate(split, label_type=label_type, prefix='cross/')

                for k, v in results.items():
                    tag = f"{split}/{k}/{label_type}"
                    self.write_scalar(tag, v, self.epoch)
                print(f'cfg NAME {self.cfg.NAME}')

            if self.cfg.TEST.CROSS_DATASET.RETRIEVAL:
                print('Evaluate CROSS retrieval')
                results_ret = self.evaluator.evaluate_retrieval_epic()

                for k, v in results_ret.items():
                    tag = f"{split}/{k}/{label_type}"
                    self.write_scalar(tag, v, self.epoch)
                print(f'cfg NAME {self.cfg.NAME}')


            return list(results.values())[0]

    # TrainerXEpic
    @torch.no_grad()
    def test_cross_egtea(self, split='test', accuracy=0):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split == 'test':
            data_loader = self.cross_test_egtea_loader
            prefix = 'egtea/'
        else:
            data_loader = self.cross_val_egtea_loader
            prefix = 'egtea_val/'

        print(f"EGTEA {split} CROSS DATASET Evaluate")
        label_types = None
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, narration_id = self.parse_batch_test(batch)
            output = self.model_inference(input, test_egtea=True)

            if isinstance(output, dict):
                label_types = list(output.keys()) if label_types is None else label_types
                # breakpoint()
                for k in output.keys():
                    if len(output) == 1 and not isinstance(label, dict):
                        self.evaluator.process(output[k], label, label_type=k)
                    else:
                        self.evaluator.process(output[k], label[k], label_type=k)
            else:
                self.evaluator.process(output, label)

        self.evaluator.synchronize_between_processes()

        if label_types is None:
            results = self.evaluator.evaluate(split, prefix=prefix, accuracy_old=accuracy, base_novel=False)

            for k, v in results.items():
                tag = f"{split}/{k}"
                self.write_scalar(tag, v, self.epoch)

            print(f'EGTEA CROSS DATASET cfg NAME {self.cfg.NAME}')
            return list(results.values())[0]
        else:
            for label_type in label_types:
                results = self.evaluator.evaluate(split, label_type=label_type, prefix=prefix, base_novel=False)

                for k, v in results.items():
                    tag = f"{split}/{k}/{label_type}"
                    self.write_scalar(tag, v, self.epoch)
                print(f'EGTEA cfg NAME {self.cfg.NAME}')

            return list(results.values())[0]

    # TrainerXEpic
    def model_inference(self, input, test=False, **kwargs):
        return self.model(input, test=test, **kwargs)

    # TrainerXEpic
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        narration_id = batch['narration_id']

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, narration_id


class TrainerXU(SimpleTrainer):
    """A base trainer using unlabeled data only."""
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_u)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_u):
            input = batch["img"]
            label = batch["label"]
            domain = batch["domain"]
