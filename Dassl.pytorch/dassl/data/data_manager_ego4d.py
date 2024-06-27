import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import pickle
import datetime
import time
from collections import defaultdict

from dassl.utils import read_image

from .datasets import build_dataset, DatasetBase
# from .samplers import build_sampler
# from .transforms import INTERPOLATION_MODES, build_transform
# from .utils.dist_utils import get_world_size, get_rank

import itertools
import os

from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)

# from .build import DATASET_REGISTRY
from datasets.ego4d_helper import *
from datasets.ego4d_helper_feature_extraction import clip_recognition_dataset_feature_extraction
# from . import ptv_dataset_helper
# from ..utils import logging, video_transformer

# logger = logging.get_logger(__name__)


from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from typing import Dict, Any

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing



def build_data_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test", "cross_eval"]
    if split in ["train"]:
        # dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
    elif split in ["val", "cross_eval"]:
        # dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        # dataset_name = cfg.TEST.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # def get_collate(key):
    #     if key == "detection":
    #         return detection_collate
    #     elif key == "short_term_anticipation":
    #         return sta_collate
    #     else:
    #         return None

    # Construct the dataset
    if split == 'cross_eval':
        dataset = build_dataset(cfg, split, dataset_name=cfg.TEST.CROSS_DATASET.DATASET_NAME)
    else:
        dataset = build_dataset(cfg, split)
    # Create a sampler for multi-process training

    sampler = None
    if not cfg.FBLEARNER:
        # Create a sampler for multi-process training
        if hasattr(dataset, "sampler"):
            sampler = dataset.sampler
        elif cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=None,
    )
    return loader, dataset


class Ego4DBaseDataset(DatasetBase):
    def __init__(self, cfg, train_x=None, train_u=None, val=None, test=None):
        self.cfg = cfg
        self.label_type = cfg.DATASET.LABEL_TYPE
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(train_x)
        print('SUPER INIT', self._num_classes,  flush=True)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)
        self._subclassnames = None
        self._classnames_detic = None

        if self.label_type == 'all':
            self.train_class_counts, self.dist_splits = self.create_lt_splits_all(cfg, train_x)
            self.train_class_counts, self.dist_splits_novel_base = self.create_base_novel_splits_all(cfg, train_x)
            self.n_test_classes = {}
            for k,v in self._classnames.items():
                self.n_test_classes[k] = len(v)
            self.test_classes=  self._classnames
            # self.n_test_classes = len(self.train_class_counts)
        else:
            self.train_class_counts, self.dist_splits = self.create_lt_splits(cfg, train_x)
            self.train_class_counts, self.dist_splits_novel_base = self.create_base_novel_splits(cfg, train_x)
            self.n_test_classes = len(self.train_class_counts)



    def get_num_classes(self, data_source=None):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        if self.label_type == 'noun':
            return 521 # 478 - train
        elif self.label_type == 'verb':
            return 117 # 116 - train
        elif self.label_type == 'all':
            dict_labels = {'noun': 521, 'verb': 117, 'action': 7099}
            output = {}
            for label_type in ['noun', 'verb', 'action']:
                if label_type not in self.cfg.DATASET.LABEL_SUBTYPES: continue
                output[label_type] = dict_labels[label_type]
            return output
        return 7099

    def get_lab2cname(self, data_source):
        if self.label_type == 'noun':
            return data_source.mappings['mapping_nouns'], data_source.mappings['classnames_nouns']
        elif self.label_type == 'verb':
            return data_source.mappings['mapping_verbs'], data_source.mappings['classnames_verbs']
        elif self.label_type == 'all':
            mappings = {}
            classnames = {}
            for label_type in ['noun', 'verb', 'action']:
                if label_type not in self.cfg.DATASET.LABEL_SUBTYPES: continue
                # output[label_type] = dict_labels[label_type]

                mappings[label_type] = data_source.mappings[f'mapping_{label_type}s']
                classnames[label_type] = data_source.mappings[f'classnames_{label_type}s']
            # mappings['noun'] = data_source.mappings['mapping_nouns']
            # classnames['noun'] = data_source.mappings['classnames_nouns']
            # mappings['action'] = data_source.mappings['mapping_actions']
            # classnames['action'] = data_source.mappings['classnames_actions']
            return mappings, classnames
        else:
            return None, None


    def create_lt_splits(self, cfg, data_source):
        if not cfg.TEST.LT_EVAL and not cfg.TRAINER.BALANCED_CE:
            return None, None

        train_class_counts = defaultdict(int)
        dist_splits = defaultdict(list)
        # breakpoint()
        total_sum = 0
        for item in data_source.dataset._labeled_videos:
            item = item[1]
            train_class_counts[item[f'{self.label_type}_label']] += 1

        num_classes = self.get_num_classes()
        if len(train_class_counts) < num_classes:
            for class_idx in range(num_classes):
                if class_idx in train_class_counts:
                    total_sum += train_class_counts[class_idx]
                    continue
                else:
                    train_class_counts[class_idx] = 0

        head_split = []
        tail_split = []
        fewshot_split = []
        cumulative_sum = 0
        half_total_sum = total_sum / 2
        for k, v in sorted(train_class_counts.items(), key=lambda x: -x[1]):
            if cumulative_sum <= half_total_sum:
                head_split.append(k)
            elif v <= 20:
                fewshot_split.append(k)
            else:
                tail_split.append(k)
            cumulative_sum += v

        print('train_class_counts', train_class_counts, flush=True)
        print('FEW_SPLIT', fewshot_split, flush=True)
        print('TAIL_SPLIT', tail_split, flush=True)
        print('HEAD_SPLIT', head_split, flush=True)

        return train_class_counts, [fewshot_split, tail_split, head_split]

    def create_lt_splits_all(self, cfg, data_source):
        if not cfg.TEST.LT_EVAL and not cfg.TRAINER.BALANCED_CE:
            return None, None

        num_classes = self.get_num_classes()
        train_class_counts = defaultdict(int)
        dist_splits = defaultdict(list)
        # breakpoint()
        total_sum = {}
        head_split = {} # []
        tail_split = {} #[]
        fewshot_split = {} #[]
        train_class_counts = {}
        for label_type in ['noun', 'verb', 'action']:
            if label_type not in cfg.DATASET.LABEL_SUBTYPES: continue
            train_class_counts[label_type] = {}
            total_sum[label_type] = 0

            for item in data_source.dataset._labeled_videos:
                item = item[1]
                train_class_counts[label_type][item[f'{label_type}_label']] += 1

            if len(train_class_counts[label_type]) < num_classes[label_type]:
                for class_idx in range(num_classes[label_type]):
                    if class_idx in train_class_counts[label_type]:
                        total_sum[label_type] += train_class_counts[label_type][class_idx]
                        continue
                    else:
                        train_class_counts[label_type][class_idx] = 0

            head_split[label_type] = []
            tail_split[label_type] = []
            fewshot_split[label_type] = []
            cumulative_sum = 0
            half_total_sum = total_sum[label_type] / 2
            for k, v in sorted(train_class_counts[label_type].items(), key=lambda x: -x[1]):
                if cumulative_sum <= half_total_sum:
                    head_split[label_type].append(k)
                elif v <= 20:
                    fewshot_split[label_type].append(k)
                else:
                    tail_split[label_type].append(k)
                cumulative_sum += v

        print('train_class_counts', train_class_counts, flush=True)
        print('FEW_SPLIT', fewshot_split, flush=True)
        print('TAIL_SPLIT', tail_split, flush=True)
        print('HEAD_SPLIT', head_split, flush=True)

        return train_class_counts, [fewshot_split, tail_split, head_split]

    def create_base_novel_splits(self, cfg, data_source):
        if not cfg.TEST.BASE_NOVEL_EVAL:
            return None, None

        classname2label = {v:k for k,v in self._lab2cname.items()}
        print(len(classname2label), len(self._lab2cname))
        assert len(classname2label) == len(self._lab2cname)


        def _fix_class(class_name):
            if class_name.startswith('nut_'):
                if 'food' in class_name:
                    class_name = 'nut'
                elif 'tool' in class_name:
                    class_name = 'nut_tool'
            elif class_name.startswith('chip_'):
                if 'food' not in class_name:
                    if "wood\'" in class_name:
                        class_name = 'chip_wood'
                    else:
                        class_name = 'chip_metal'
            elif class_name.startswith('bat_'):
                if 'tool' in class_name:
                    class_name = 'bat_tool'
                elif 'sports' in class_name:
                    class_name = 'bat_sports'
            elif class_name.startswith('pot'):
                if 'planter' in class_name:
                    class_name = 'planter'
            class_name = class_name.split('(')[0]
            return class_name

        # base_classnames = []
        # with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR + f'ego_epic_common_{self.label_type}.txt'), 'r') as f:
        #     for line in f:
        #         base_classnames.append(line.strip())
        #
        # novel_classnames = []
        # with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR + f'ego_epic_non_common_{self.label_type}.txt'), 'r') as f:
        #     for line in f:
        #         novel_classnames.append(line.strip())
        #
        # base_split = [classname2label[_fix_class(base_name)] for base_name in base_classnames]
        # novel_split = [classname2label[_fix_class(novel_name)] for novel_name in novel_classnames]
        #
        # print('BASE_SPLIT', base_split, flush=True)
        # print('NOVEL_SPLIT', novel_split, flush=True)

        shared_classnames_exact = []  # epic_ego_common_exact_noun.txt
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'ego_epic_common_exact_{self.label_type}.txt'), 'r') as f:
            for line in f:
                shared_classnames_exact.append(line.strip())

        unique_classnames_exact = []
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,  f'ego_epic_non_common_exact_{self.label_type}.txt'),'r') as f:
            for line in f:
                unique_classnames_exact.append(line.strip())

        shared_split_exact = [classname2label[_fix_class(base_name)] for base_name in shared_classnames_exact]
        unique_split_exact = [classname2label[_fix_class(novel_name)] for novel_name in unique_classnames_exact]

        print('SHARED_EXACT_SPLIT', len(shared_split_exact), shared_split_exact, flush=True)
        print('UNIQUE_EXACT_SPLIT', len(unique_split_exact), unique_split_exact, flush=True)

        shared_classnames_semantic = []  # epic_ego_common_exact_noun.txt
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'ego_epic_common_semantic_{self.label_type}.txt'), 'r') as f:
            for line in f:
                shared_classnames_semantic.append(line.strip())

        shared_classnames_semantic_wo_exact = []  # epic_ego_common_exact_noun.txt
        for class_name in shared_classnames_semantic:
            if class_name in shared_classnames_exact: continue
            else:
                shared_classnames_semantic_wo_exact.append(class_name)

        unique_classnames_semantic = []
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,  f'ego_epic_non_common_semantic_{self.label_type}.txt'),'r') as f:
            for line in f:
                unique_classnames_semantic.append(line.strip())

        shared_split_semantic = [classname2label[_fix_class(base_name)] for base_name in shared_classnames_semantic]
        unique_split_semantic = [classname2label[_fix_class(novel_name)] for novel_name in unique_classnames_semantic]
        shared_split_semantic_wo_exact = [classname2label[_fix_class(novel_name)] for novel_name in shared_classnames_semantic_wo_exact]

        print('SHARED_SEMANTIC_SPLIT', len(shared_split_semantic), shared_split_semantic, flush=True)
        print('UNIQUE_SEMANTIC_SPLIT', len(unique_split_semantic), unique_split_semantic, flush=True)
        print('shared_split_semantic_wo_exact', len(shared_split_semantic_wo_exact), shared_split_semantic_wo_exact, flush=True)

        train_class_counts = defaultdict(int)
        for item in data_source.dataset._labeled_videos:
            item = item[1]
            train_class_counts[item[f'{self.label_type}_label']] += 1

        num_classes = self.get_num_classes()
        if len(train_class_counts) < num_classes:
            for class_idx in range(num_classes):
                if class_idx in train_class_counts:
                    continue
                else:
                    train_class_counts[class_idx] = 0

        return train_class_counts, [shared_split_exact, unique_split_exact, shared_split_semantic, unique_split_semantic, shared_split_semantic_wo_exact]

    def create_base_novel_splits_all(self, cfg, data_source):
        if not cfg.TEST.BASE_NOVEL_EVAL:
            return None, None

        shared_split_exact = {}
        unique_split_exact = {}
        shared_split_semantic = {}
        unique_split_semantic = {}
        shared_split_semantic_wo_exact = {}
        train_class_counts = {}
        num_classes = self.get_num_classes()
        assert isinstance(self._lab2cname, dict)
        for label_type in ['noun', 'verb']:
            if label_type not in cfg.DATASET.LABEL_SUBTYPES: continue

            classname2label = {v:k for k,v in self._lab2cname[label_type].items()}
            print(len(classname2label), len(self._lab2cname[label_type]))
            assert len(classname2label) == len(self._lab2cname[label_type])


            def _fix_class(class_name):
                if class_name.startswith('nut_'):
                    if 'food' in class_name:
                        class_name = 'nut'
                    elif 'tool' in class_name:
                        class_name = 'nut_tool'
                elif class_name.startswith('chip_'):
                    if 'food' not in class_name:
                        if "wood\'" in class_name:
                            class_name = 'chip_wood'
                        else:
                            class_name = 'chip_metal'
                elif class_name.startswith('bat_'):
                    if 'tool' in class_name:
                        class_name = 'bat_tool'
                    elif 'sports' in class_name:
                        class_name = 'bat_sports'
                elif class_name.startswith('pot'):
                    if 'planter' in class_name:
                        class_name = 'planter'
                class_name = class_name.split('(')[0]
                return class_name


            shared_classnames_exact = []  # epic_ego_common_exact_noun.txt
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'ego_epic_common_exact_{label_type}.txt'), 'r') as f:
                for line in f:
                    shared_classnames_exact.append(line.strip())

            unique_classnames_exact = []
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,  f'ego_epic_non_common_exact_{label_type}.txt'),'r') as f:
                for line in f:
                    unique_classnames_exact.append(line.strip())

            shared_split_exact[label_type] = [classname2label[_fix_class(base_name)] for base_name in shared_classnames_exact]
            unique_split_exact[label_type] = [classname2label[_fix_class(novel_name)] for novel_name in unique_classnames_exact]

            print('SHARED_EXACT_SPLIT', len(shared_split_exact[label_type]), shared_split_exact[label_type], flush=True)
            print('UNIQUE_EXACT_SPLIT', len(unique_split_exact[label_type]), unique_split_exact[label_type], flush=True)

            shared_classnames_semantic = []  # epic_ego_common_exact_noun.txt
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'ego_epic_common_semantic_{label_type}.txt'), 'r') as f:
                for line in f:
                    shared_classnames_semantic.append(line.strip())

            shared_classnames_semantic_wo_exact = []  # epic_ego_common_exact_noun.txt
            for class_name in shared_classnames_semantic:
                if class_name in shared_classnames_exact: continue
                else:
                    shared_classnames_semantic_wo_exact.append(class_name)

            unique_classnames_semantic = []
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,  f'ego_epic_non_common_semantic_{label_type}.txt'),'r') as f:
                for line in f:
                    unique_classnames_semantic.append(line.strip())

            shared_split_semantic[label_type] = [classname2label[_fix_class(base_name)] for base_name in shared_classnames_semantic]
            unique_split_semantic[label_type] = [classname2label[_fix_class(novel_name)] for novel_name in unique_classnames_semantic]
            shared_split_semantic_wo_exact[label_type] = [classname2label[_fix_class(novel_name)] for novel_name in shared_classnames_semantic_wo_exact]

            print('SHARED_SEMANTIC_SPLIT', len(shared_split_semantic[label_type]), shared_split_semantic[label_type], flush=True)
            print('UNIQUE_SEMANTIC_SPLIT', len(unique_split_semantic[label_type]), unique_split_semantic[label_type], flush=True)
            print('shared_split_semantic_wo_exact', len(shared_split_semantic_wo_exact[label_type]), shared_split_semantic_wo_exact[label_type], flush=True)

            train_class_counts[label_type] = defaultdict(int)
            for item in data_source.dataset._labeled_videos:
                item = item[1]
                train_class_counts[label_type][item[f'{label_type}_label']] += 1

            if len(train_class_counts[label_type]) < num_classes[label_type]:
                for class_idx in range(num_classes[label_type]):
                    if class_idx in train_class_counts[label_type]:
                        continue
                    else:
                        train_class_counts[label_type][class_idx] = 0

        return train_class_counts, [shared_split_exact, unique_split_exact, shared_split_semantic, unique_split_semantic, shared_split_semantic_wo_exact]

class Ego4DDataManagerCrossEval:

    def __init__(
        self,
        cfg,
    ):

        # Build val_loader
        val_loader, dataset_val = build_data_loader(
            cfg,
            split='cross_eval'
        )

        self.train_loader_u = None
        dataset = Ego4DBaseDataset(cfg=cfg, train_x=dataset_val, val=dataset_val, test=dataset_val)
        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = val_loader
        self.val_loader = val_loader
        self.test_loader = val_loader
        # print(dataset._lab2cname)
        self._lab2cname = dataset.lab2cname
        # print('CROSS EVAL LAB2CNAME', [(k, v) for k,v in sorted(self._lab2cname.items(), key=lambda x: x[0])])

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        if isinstance(self.num_classes, dict):
            for k, v in self.num_classes.items():
                table.append([f"# classes {k}", f"{v:,}"])
        else:
            table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))

class Ego4DDataManager:

    def __init__(
        self,
        cfg,
        cross_cfg=None
        # custom_tfm_train=None,
        # custom_tfm_test=None,
        # dataset_wrapper=None
    ):
        # Load dataset
        # dataset = build_dataset(cfg)
        # self._lab2cname = dataset.lab2cname

        # # Build transform
        # if custom_tfm_train is None:
        #     tfm_train = build_transform(cfg, is_train=True)
        # else:
        #     print("* Using custom transform for training")
        #     tfm_train = custom_tfm_train

        # if custom_tfm_test is None:
        #     tfm_test = build_transform(cfg, is_train=False)
        # else:
        #     print("* Using custom transform for testing")
        #     tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x, dataset_train = build_data_loader(
            cfg,
            split='train'
        )

        # Build val_loader
        val_loader, dataset_val = build_data_loader(
            cfg,
            split='val'
        )

        # Build test_loader
        if cfg.TEST.LOADER == 'val':
            test_loader, dataset_test = val_loader, dataset_val
        else:
            test_loader, dataset_test = build_data_loader(
                cfg,
                split='test'
            )
        self.train_loader_u = None
        dataset = Ego4DBaseDataset(cfg=cfg, train_x=dataset_train, val=dataset_val, test=dataset_test)
        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.val_loader = val_loader
        self.test_loader = test_loader
        # print(dataset._lab2cname)
        self._lab2cname = dataset.lab2cname
        # print('LAB2CNAME', [(k, v) for k,v in sorted(self._lab2cname.items(), key=lambda x: x[0])])

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        if isinstance(self.num_classes, dict):
            for k, v in self.num_classes.items():
                table.append([f"# classes {k}", f"{v:,}"])
        else:
            table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))

class CenterClipVideoSampler(ClipSampler):
    """
    Samples just a single clip from the center of the video (use for testing)
    """

    def __init__(
        self, clip_duration: float
    ) -> None:
        super().__init__(clip_duration)

    def __call__(
        self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]
    ) -> ClipInfo:

        clip_start_sec = video_duration / 2 - self._clip_duration / 2

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            0,
            0,
            True,
        )

@DATASET_REGISTRY.register()
class Ego4DRecognitionWrapper(TorchDataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
            "cross_eval"
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        # Random Clip Sampler  Returns:
        #             clip_info (ClipInfo): includes the clip information of (clip_start_time,
        #             clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
        #             clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        # Uniform Clip Sampler Returns:
        #             clip_info: (ClipInfo): includes the clip information (clip_start_time,
        #             clip_end_time, clip_index, aug_index, is_last_clip), where the times are in
        #             seconds and is_last_clip is False when there is still more of time in the video
        #             to be sampled.
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)


        mode_ = 'test_unannotated' if mode=='test' else mode
        mode_ = 'val' if mode=='cross_eval' else mode_

        # data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}_w_narrations.json')

        if mode in ['train', 'cross_eval']:
            self.dataset, self.mappings = clip_recognition_dataset(
                data_path=data_path,
                clip_sampler=clip_sampler,
                video_sampler=sampler,
                decode_audio=False,
                transform=self._make_transform(mode, cfg),
                video_path_prefix=self.cfg.DATA.PATH_PREFIX,
                classnames_and_mapping=True,
                use_features=self.cfg.DATALOADER.USE_EXTRACTED_FEATURES,
                split='val' if mode=='cross_eval' else mode,
                cfg=self.cfg
            )
        else:
            self.mappings = None
            self.dataset = clip_recognition_dataset(
                data_path=data_path,
                clip_sampler=clip_sampler,
                video_sampler=sampler,
                decode_audio=False,
                transform=self._make_transform(mode, cfg),
                video_path_prefix=self.cfg.DATA.PATH_PREFIX,
                use_features=self.cfg.DATALOADER.USE_EXTRACTED_FEATURES,
                split=mode,
                cfg=self.cfg
            )

        self.dataset_full_frames = None
        if self.cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES_ONLY:
            self.dataset_full_frames = clip_recognition_dataset(
                data_path=data_path,
                clip_sampler=clip_sampler,
                video_sampler=sampler,
                decode_audio=False,
                transform=self._make_transform(mode, cfg),
                video_path_prefix=self.cfg.DATA.PATH_PREFIX,
                use_features=False,
                split=mode,
                cfg=self.cfg
            )

        if self.cfg.DATALOADER.NUM_WORKERS == 0:
            self._dataset_iter = itertools.chain.from_iterable(
                itertools.repeat(iter(self.dataset), self.cfg.OPTIM.MAX_EPOCH)
            )
        else:
            self._dataset_iter = itertools.chain.from_iterable(
                itertools.repeat(iter(self.dataset), 2)
            )

# int(pts - start_pts) * time_base

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):

        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x/255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + random_scale_crop_flip(mode, cfg)
                        + [uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["video_name"]) + "_" + str(x["video_index"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        if self.dataset_full_frames is not None:
            out = self.dataset_full_frames.get_item(value["video_index"])
            value['img'] = out[0][0].transpose(0, 1)
        return value

    def __len__(self):
        return self.dataset.num_videos


@DATASET_REGISTRY.register()
class Ego4DRecognitionWrapperFeatureExtraction(TorchDataset):
    def __init__(self, cfg, mode, save_root, seed=0):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        # if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
        #     sampler = DistributedSampler

        clip_sampler_type = "uniform"
        # Random Clip Sampler  Returns:
        #             clip_info (ClipInfo): includes the clip information of (clip_start_time,
        #             clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
        #             clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.

        # Uniform Clip Sampler Returns:
        #             clip_info: (ClipInfo): includes the clip information (clip_start_time,
        #             clip_end_time, clip_index, aug_index, is_last_clip), where the times are in
        #             seconds and is_last_clip is False when there is still more of time in the video
        #             to be sampled.
        clip_duration = (self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
                        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode == 'test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')

        self.dataset = clip_recognition_dataset_feature_extraction(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            decode_audio=False,
            # transform=self._make_transform('test', cfg),
            transform=None if cfg.DATALOADER.HAND_CROPS else self._make_transform('test', cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
            save_root=save_root,
            split=mode,
            seed=self.cfg.SEED,
            cfg=cfg,
        )

        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            # UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x / 255.0),
                            Resize((cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE)),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["full_feature_path"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


def read_feature(path):
    if path.endswith('npz') or path.endswith('npy'):
        return np.load(path)
    if path.endswith('pkl') or path.endswith('pickle'):
        with open(path, 'wb') as f:
            return pickle.load(f)

