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
import gc
import pandas as pd
import decord

from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform
from datasets.egtea import get_raw_item
# from .utils.dist_utils import get_world_size, get_rank

import torchvision.transforms as transforms
from lavila.transforms import Permute, SpatialCrop, TemporalCrop
import torchvision.transforms._transforms_video as transforms_video



def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    lab2cname=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
        is_train=is_train
    )

    # if cfg.DISTRIBUTED:
    #     sampler = DistributedSamplerWrapper(
    #         sampler=sampler,
    #         num_replicas=get_world_size(),
    #         rank=get_rank(),
    #         shuffle=is_train)

    if dataset_wrapper is None:
        data_loader = torch.utils.data.DataLoader(
            data_source,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train, lab2cname=lab2cname),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
        )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)
        self._lab2cname = dataset.lab2cname

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
            lab2cname=self._lab2cname,
        )

        # Build train_loader_u
        train_loader_u = None

        # Build val_loader
        val_loader = None
        if dataset.val:
            if cfg.DATALOADER.EGOCLIP.EVAL:
                batch_size = 128
            else:
                batch_size = cfg.DATALOADER.TEST.BATCH_SIZE
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=batch_size,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
                lab2cname=self._lab2cname
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
            lab2cname=self._lab2cname
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.text_val_dataloader = None
        if cfg.TEST.RETRIEVAL:
            self.text_val_dataloader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.text_val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=None,
                is_train=False,
                dataset_wrapper=TextWrapper,
                lab2cname=self._lab2cname
            )

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
            for k,v in self.num_classes.items():
                table.append([f"# classes {k}", f"{v:,}"])
        else:
            table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))

class DataManagerCrossEval:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None,
        egtea=False,
        batch_size=0,
    ):
        # Load dataset
        if egtea:
            dataset = build_dataset(cfg, dataset_name="EGTEA")
        else:
            dataset = build_dataset(cfg, dataset_name=cfg.TEST.CROSS_DATASET.DATASET_NAME)
        self._lab2cname = dataset.lab2cname

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x

        # Build train_loader_u
        train_loader_u = None
        batch_size = batch_size if batch_size != 0 else cfg.DATALOADER.TEST.BATCH_SIZE

        # Build val_loader
        val_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.val,
            batch_size=batch_size,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
            lab2cname=self._lab2cname
        )

        # Build test_loader
        test_loader = val_loader
        if egtea:
            test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test,
                batch_size=batch_size,
                tfm=tfm_test,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
                lab2cname=self._lab2cname
            )

        train_loader_x = val_loader

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.text_val_dataloader = None
        if cfg.TEST.CROSS_DATASET.RETRIEVAL:
            self.text_val_dataloader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.text_val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=TextWrapper,
                lab2cname=self._lab2cname
            )

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
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, lab2cname=None):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


class TextWrapper(TorchDataset):
    def __init__(self, cfg, data_source, transform=None, is_train=False, transform_crops=None, meta_data_only=False, lab2cname=None):
        self.cfg = cfg
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, item):
        return self.data_source[item]


class DatasetSegmentWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, transform_crops=None, meta_data_only=False, lab2cname=None):
        self.cfg = cfg
        self.data_source = data_source
        self.meta_data_only = meta_data_only
        self.transform = transform  # accept list (tuple) as input
        self.toTensor = transforms.ToTensor()
        self.is_train = is_train
        self.label_type = self.cfg.DATASET.LABEL_TYPE
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.use_extracted_DINO_features_only = cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES_ONLY
        self.use_objects_features = cfg.DATALOADER.USE_OBJECTS_FEATURES
        self.use_dino_features = cfg.DATALOADER.USE_DINO_FEATURES
        self.use_dino_features2 = cfg.DATALOADER.USE_DINO_FEATURES2
        self.use_lavila_features = cfg.DATALOADER.LAVILA_FEATURES
        self.load_all_features_at_once = cfg.DATALOADER.LOAD_ALL_FEATURES_AT_ONCE
        self.features = None
        self.object_features = None
        self.features_dim = cfg.DATALOADER.FEATURES_DIM
        self.detic_crops = cfg.DATALOADER.DETIC.CROPS
        self.lab2cname = lab2cname
        self.use_videos = cfg.DATALOADER.EPIC_VIDEOS
        if self.use_videos:
            self.path_handler = VideoPathHandler()
            self._decoder = 'pyav'
            self._decode_audio = False
            image_dir = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            video_info_csv = pd.read_csv(os.path.join(image_dir, 'annotations', 'EPIC_100_video_info.csv'))
            self.video_info = {}
            # duration      1652.152817
            # fps              59.94006
            # resolution      1920x1080
            for row in video_info_csv.iterrows():
                self.video_info[row[1].video_id] = {'fps': row[1].fps,
                                               'duration': row[1].duration,
                                               'resolution': row[1].resolution}


        if self.use_extracted_features and self.load_all_features_at_once:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            if is_train:
                if not self.use_extracted_DINO_features_only:
                    full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'train.pkl')  # split narration_id
                    with open(full_feature_path, 'rb') as f:
                        # print('load train features')
                        time_start = time.time()
                        self.features = pickle.load(f)
                        elapsed = round(time.time() - time_start)
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(f"Train features load Elapsed: {elapsed}")
                else:
                    self.features = None

                if self.use_dino_features:
                    if cfg.DATALOADER.FEATURES_NAME == cfg.DATALOADER.FEATURES_NAME_DINO:
                        self.features_dino = self.features
                        print(f"Train DINO features load Elapsed: {0}")
                    else:
                        full_feature_path_dino = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                              'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO,
                                                              'train.pkl')  # split narration_id
                        with open(full_feature_path_dino, 'rb') as f_dino:
                            # print('load train features')
                            time_start = time.time()
                            self.features_dino = pickle.load(f_dino)
                            elapsed = round(time.time() - time_start)
                            elapsed = str(datetime.timedelta(seconds=elapsed))
                            print(f"Train DINO features load Elapsed: {elapsed}")
                if self.use_dino_features2:
                    if cfg.DATALOADER.FEATURES_NAME_DINO == cfg.DATALOADER.FEATURES_NAME_DINO2:
                        self.features_dino2 = self.features_dino
                        print(f"Train DINO features load Elapsed: {0}")
                    else:
                        full_feature_path_dino = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                              'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO2,
                                                              'train.pkl')  # split narration_id
                        with open(full_feature_path_dino, 'rb') as f_dino:
                            # print('load train features')
                            time_start = time.time()
                            self.features_dino2 = pickle.load(f_dino)
                            elapsed = round(time.time() - time_start)
                            elapsed = str(datetime.timedelta(seconds=elapsed))
                            print(f"Train DINO features load Elapsed: {elapsed}")
            else:
                if not self.use_extracted_DINO_features_only:
                    full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'validation.pkl')  # split narration_id
                    with open(full_feature_path, 'rb') as f:
                        # print('load val features')
                        time_start = time.time()
                        self.features = pickle.load(f)
                        elapsed = round(time.time() - time_start)
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(f"Val features load Elapsed: {elapsed}")
                else:
                    self.features = None

                if self.use_dino_features:
                    if cfg.DATALOADER.FEATURES_NAME == cfg.DATALOADER.FEATURES_NAME_DINO:
                        self.features_dino = self.features
                        print(f"Val DINO features load Elapsed: {0}")
                    else:
                        full_feature_path_dino = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                              'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO,
                                                              'validation.pkl')  # split narration_id
                        with open(full_feature_path_dino, 'rb') as f_dino:
                            # print('load train features')
                            time_start = time.time()
                            self.features_dino = pickle.load(f_dino)
                            elapsed = round(time.time() - time_start)
                            elapsed = str(datetime.timedelta(seconds=elapsed))
                            print(f"Val DINO features load Elapsed: {elapsed}")

                if self.use_dino_features2:
                    if cfg.DATALOADER.FEATURES_NAME_DINO2 == cfg.DATALOADER.FEATURES_NAME_DINO:
                        self.features_dino2 = self.features_dino
                        print(f"Val DINO features load Elapsed: {0}")
                    else:
                        full_feature_path_dino = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                              'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO2,
                                                              'validation.pkl')  # split narration_id
                        with open(full_feature_path_dino, 'rb') as f_dino:
                            # print('load train features')
                            time_start = time.time()
                            self.features_dino2 = pickle.load(f_dino)
                            elapsed = round(time.time() - time_start)
                            elapsed = str(datetime.timedelta(seconds=elapsed))
                            print(f"Val DINO features load Elapsed: {elapsed}")
        elif self.use_extracted_features:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            if is_train:
                prefix = 'train_%s.npy'
            else:
                prefix = 'validation_%s.npy'

            # local_path = root + f'segments_npy/{split_file}_{narration_id}.npy'
            self.full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                  'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'segments_npy',
                                                  prefix)  # split narration_id
            self.features = None
            if self.use_dino_features:
                self.full_feature_path_dino = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                           'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO, 'segments_npy',
                                                           prefix)
                self.features_dino = None
            if self.use_dino_features2:
                self.full_feature_path_dino2 = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                            'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO2, 'segments_npy',
                                                            prefix)
                self.features_dino2 = None




        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.frames_per_seg = cfg.DATALOADER.FRAMES_PER_SEGMENT # TODO: add to config
        crop_file_name = f'hand_thd{cfg.DATALOADER.CROPPER.HAND_THS}_obj_thd{cfg.DATALOADER.CROPPER.OBJ_THS}{"_ONLY_inter_obj" if cfg.DATALOADER.CROPPER.ONLY_INTERACTED_OBJ else ""}{"_with_HANDS" if cfg.DATALOADER.CROPPER.WITH_HANDS else ""}{cfg.DATALOADER.CROPPER.TAG}'
        self.path_to_save_random_crops = os.path.join(os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)), 'annotations', 'hand_crops', 'croped_random_frames', f'{crop_file_name}')
        print('crop path', self.path_to_save_random_crops )
        os.makedirs(self.path_to_save_random_crops, exist_ok=True)
        # self.path_to_save_random_crops = f'{cfg.DATASET.DETECTION_ROOT}/croped_random_frames/{crop_file_name}/'

        self.hand_crops = cfg.DATALOADER.HAND_CROPS
        self.return_full_frame = cfg.DATALOADER.CROPPER.FULL_FRAME
        self.mask_background = cfg.DATALOADER.CROPPER.BLACK_CROP


        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)


        self.detic_lab2subset_lab = None

    def __len__(self):
        return len(self.data_source)

    def get_item(self, idx):
        # tmp_only_meta_data = self.meta_data_only
        output = self.__getitem__(idx, meta_data_only=False)
        # self.meta_data_only = tmp_only_meta_data
        return output

    def _uniform_frame_generator(self, start_frame, end_frame):
        if self.frames_per_seg <= -1:  # all frames
            uniform_frames = range(start_frame, end_frame)
        else:
            uniform_frames, step = np.linspace(start_frame, end_frame - 1, self.frames_per_seg, endpoint=False, retstep=True)
            if self.is_train:
                # shift = np.random.choice(np.arange(int(step))) if int(step) else 0
                if int(step):
                    uniform_frames = [int(i) + np.random.choice(np.arange(int(step)))  for i in uniform_frames]
                else:
                    uniform_frames = [int(i) for i in uniform_frames]
            else:
                # shift = 0
                uniform_frames = [int(i + step / 2) for i in uniform_frames]
            # uniform_frames = [int(i) + shift for i in uniform_frames]
        return uniform_frames


    def __getitem__(self, idx, meta_data_only=None):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]
        path = item['path']
        video_path = item['video_path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']


        feature_path  = item['feature_path']

        output = {
            "domain": 0,
            "impath": '',
            "index": idx,
            "narration_id": item['narration_id'],
            'subclass_label': item['subclass_label'] if 'subclass_label' in item else -1,
            "narration": item['narration']
        }

        if self.label_type in ['all', 'all_w_narrations']:
            for label_type in ['noun', 'verb', 'action']:
                output[f'{label_type}_label'] = item[f'{label_type}_class']

        else:
            output['label'] = item[f'{self.label_type}_class']

        if meta_data_only is None and self.meta_data_only:
            return output

        # TODO: check what is faster to collect images and then apply transform to all of them at the same time,
        # or apply transform to each image separately and then concatenate them
        # if self.hand_crops:
            # video_detections = load_detections(item['dection_path'])

        start_frame_feature_offset = 0
        if self.cfg.DATALOADER.EXTRACT_FEAT_WITH_OFFSET_SECS > 0:
            if self.cfg.DATALOADER.USE_FEAT_WITH_OFFSET_SECS > 0:
                offset_frames = int(self.cfg.DATALOADER.USE_FEAT_WITH_OFFSET_SECS * item['fps'])

                start_frame = max(1, start_frame - offset_frames)
                end_frame = min(item["total_frame_number"], end_frame + offset_frames)
                start_frame_feature_offset = self.cfg.DATALOADER.EXTRACT_FEAT_WITH_OFFSET_SECS - self.cfg.DATALOADER.USE_FEAT_WITH_OFFSET_SECS
            else:
                offset_frames = int(self.cfg.DATALOADER.EXTRACT_FEAT_WITH_OFFSET_SECS * item['fps'])

                start_frame = max(1, start_frame - offset_frames)
                end_frame = min(item["total_frame_number"], end_frame + offset_frames)



        output_img_stack = []
        if self.frames_per_seg <= -1: # all frames
            uniform_frames = range(start_frame, end_frame)
        elif self.frames_per_seg == 0:
            return output
        else:
            uniform_frames, step = np.linspace(start_frame, end_frame-1, self.frames_per_seg, endpoint=False, retstep=True)
            if self.is_train:
                if int(step):
                    uniform_frames = [int(i) + np.random.choice(np.arange(int(step))) for i in uniform_frames]
                else:
                    uniform_frames = [int(i) for i in uniform_frames]
                # shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                uniform_frames = [int(i + step / 2) for i in uniform_frames]
                # shift = 0
            # uniform_frames = [int(i) + shift for i in uniform_frames]


        #################################################
        ################## USING FEATURES ###############
        #################################################

        if self.use_extracted_features:
            # try:
                # if self.features is not None:
            if not self.use_extracted_DINO_features_only:
                if self.features is not None:
                    features = self.features[item['narration_id']]
                else:
                    features = np.load(self.full_feature_path % item['narration_id'] )
                uniform_frames = [i - start_frame + start_frame_feature_offset for i in uniform_frames]
                if self.use_lavila_features:
                    if self.is_train:
                        random_idx = np.random.choice(np.arange(len(features)))
                        output['img'] = torch.tensor(features[random_idx])
                    else:
                        # output['img'] = torch.tensor(features.mean(0))
                        # output['img'] = torch.tensor(features[features.shape[0] // 2])
                        output['img'] = torch.tensor(features[0])
                        # output['img'] = torch.tensor(features[1] if len(features) > 1 else features[0])
                else:
                    output['img'] = torch.tensor(features[uniform_frames])
            else:
                features = None

            if self.use_dino_features:
                if self.features_dino is not None:
                    dino_features = self.features_dino[item['narration_id']]
                else:
                    dino_features = np.load(self.full_feature_path_dino % item['narration_id'])
                if self.use_extracted_DINO_features_only or len(dino_features) < len(features) or self.use_lavila_features:
                    uniform_frames_dino = self._uniform_frame_generator(0, len(dino_features))
                else:
                    uniform_frames_dino = uniform_frames
                output['dino'] = torch.tensor(dino_features[uniform_frames_dino])

            if self.use_dino_features2:
                if self.features_dino2 is not None:
                    dino_features2 = self.features_dino2[item['narration_id']]
                else:
                    dino_features2 = np.load(self.full_feature_path_dino2 % item['narration_id'])
                if len(dino_features2) < len(dino_features):
                    uniform_frames_dino = self._uniform_frame_generator(0, min(len(dino_features), len(dino_features2)))
                dino_subset = torch.tensor(dino_features[uniform_frames_dino]).unsqueeze(0)
                dino2_subset = torch.tensor(dino_features2[uniform_frames_dino]).unsqueeze(0)
                # dino_subset = output['dino'].unsqueeze(0)
                dino_subset = torch.cat([dino_subset, dino2_subset])
                # breakpoint()
                dino_dim = dino_subset.shape[-1]
                dino_subset = dino_subset.transpose(0,1).reshape(-1, dino_dim)
                output['dino'] = dino_subset

            if not self.use_extracted_DINO_features_only:
                return output


        # if self.detic_crops:
        #     detic_crops = np.load(item['detic_meta_path'], allow_pickle=True)
        #     frame_idx2start_end = detic_crops['frame_idx2start_end'].item()
        #     data_detic_crops = detic_crops['data']
        #     output['frame_idx2start_end'] = frame_idx2start_end

        #################################################
        ################## USING VIDEOS #################
        #################################################

        if self.use_videos:
            start_sec = item['start_timestamp'].split(':')
            start_sec = int(start_sec[0])*60*60 + int(start_sec[1]) * 60 + float(start_sec[2])
            end_sec = item['stop_timestamp'].split(':')
            end_sec = int(end_sec[0])*60*60 + int(end_sec[1]) * 60 + float(end_sec[2]) + 0.1
            if end_sec - start_sec < 1:
                end_sec = start_sec + 1
            # breakpoint()
            try:
                # breakpoint()
                video = self.path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,  # False
                    decoder=self._decoder,
                )
            except Exception as e:
                print(
                    "Failed to load video with error: {}; trial {}".format(e, 0)
                )
                raise EOFError
            # breakpoint()
            video_id = item['video_id']
            fps = self.video_info[video_id]['fps']
            # start_sec = start_frame / fps
            # end_sec = (end_frame / fps) + 1.0
            # {'fps': 59.9400599400599, 'duration': 389.072017, 'resolution': '1920x1080'}
            # (Pdb) clip['video'].shape
            # torch.Size([3, 111, 1080, 1920]) -> C x T x H x W
            clip = video.get_clip(start_sec, end_sec)
            video_is_null = clip is None or clip["video"] is None
            if video_is_null:
                print(f'video is null {video_path}')
                # torch.Size([61, 3, 224, 224])
                output['img'] = torch.zeros(0,1,1,1)
                return output
            video.close()
            gc.collect()
            # check the dimensionality of obtained frames
            # frame_count_expected = end_frame + 1 - start_frame
            # frame_count_expected = len(uniform_frames)
            clip_frames = clip['video']
            # try:
            #     assert clip_frames.shape[1] >= frame_count_expected
            # except Exception:
            #     print(clip_frames.shape)
            #     print(video_path)
            #     print(start_sec, end_sec)
            #     print(start_frame, end_frame, fps)
            #     raise ValueError
            # clip_frames = clip_frames[:, :frame_count_expected]
            total_frames = clip_frames.shape[1]
            if self.frames_per_seg <= -1:  # all frames
                uniform_frames = range(0, total_frames)
            else:
                uniform_frames, step = np.linspace(0, total_frames - 1, self.frames_per_seg, endpoint=False,
                                                   retstep=True)
                if self.is_train:
                    shift = np.random.choice(np.arange(int(step))) if int(step) else 0
                else:
                    shift = 0
                uniform_frames = [int(i) + shift for i in uniform_frames]

            # breakpoint()
            clip_frames = clip_frames[:, uniform_frames]
            # C x T x H x W -> T x C x H x W
            clip_frames = clip_frames.transpose(0, 1).float() / 255
            # clip_frames = clip_frames.permute(0, 3, 1, 2).float() / 255
            if self.hand_crops:
                frame_count_expected = end_frame + 1 - start_frame

                scale_width, scale_height = None, None
                width_factor, height_factor = None, None
                cropped_frames = []
                for frame_pos_idx, frame_idx in enumerate(uniform_frames):
                    frame_idx = start_frame + int(frame_pos_idx / total_frames * frame_count_expected)
                    cur_frame = clip_frames[frame_pos_idx]
                    if scale_width is None:
                        img_i = read_image(path % frame_idx)
                        scale_width = img_i.width
                        scale_height = img_i.height
                        # fix here
                        width_factor = clip_frames[0].shape[0]
                        height_factor = clip_frames[0].shape[0]

                    if frame_idx in item['crop_bb']:  # in case of extended boundaries
                        left, top, right, bottom = item['crop_bb'][frame_idx]
                        # scale bounding boxes
                        left = left / scale_width * width_factor
                        right = right / scale_width * width_factor
                        top = top / scale_height * height_factor
                        bottom = bottom / scale_height * height_factor
                        cur_frame = cur_frame[:, top:bottom, left:right]
                    if self.transform is not None:
                        cur_frame = self._transform_image(self.transform, cur_frame)
                        cropped_frames.append(cur_frame.unsqueeze(0))

                cropped_frames = torch.stack(cropped_frames)
                # cropped_frames = cropped_frames.permute(0, 3, 1, 2)
                output['img'] = cropped_frames

            else:
                # clip_frames = clip_frames.float() / 255
                if self.transform is not None:
                    # (Pdb) clip_frames.shape (after transform)
                    # torch.Size([61, 3, 224, 224])
                    clip_frames = self._transform_image(self.transform, clip_frames)

                # clip_frames = clip_frames.permute(0, 3, 1, 2)
                output['img'] =  clip_frames

            return output




        #################################################
        ################## USING FRAMES #################
        #################################################


        save_bool = False
        already_saved = False
        for frame_idx in uniform_frames:
            img_i = read_image(path % frame_idx)
            img_full = None

            if self.hand_crops:
                # save_bool = False
                # if np.random.random() > 0.5 and not already_saved:
                #     img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}.jpg')
                #     save_bool = True
                #     already_saved = True
                if self.return_full_frame:
                    img_full = img_i.copy()


                if frame_idx in item['crop_bb']: # in case of extended boundaries
                    left, top, right, bottom = item['crop_bb'][frame_idx]
                    if (right - left) > 1 and (bottom - top) > 1:
                        if isinstance(img_i, Image.Image):
                            img_i = img_i.crop(item['crop_bb'][frame_idx])
                        else:
                            img_i = img_i[:, top:bottom, left:right]

                # if save_bool:
                #     save_bool = False
                #     img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}_crop.jpg')
            #     img_i = self.cropper.render_crops(img_i, video_detections[frame_idx])
                # print(f'Get cropped frame {frame_idx}')


            if self.transform is not None:
                # breakpoint()
                img = self._transform_image(self.transform, img_i)
                output_img_stack.append(img.unsqueeze(0))
                if img_full is not None:
                    img_full = self._transform_image(self.transform, img_full)
                    output_img_stack.append(img_full.unsqueeze(0))
            else:
                raise NotImplementedError("Reading data with multiple transformations..")

        output_img_stack = torch.cat(output_img_stack, dim=0)
        output['img'] = output_img_stack
        # print(f'Get cropped frames {frame_idx}')
        # breakpoint()
        return output

    def _get_video_frames(self, video_fp, video_sec, bound_sec, boxes=None, pred=False):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, seconds = self.video_reader(video_fp[0], video_sec[0], end_second=video_sec[1],
                                                  clip_length=self.video_params['num_frames'])
                valid = 1
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0).to(torch.float32)
                valid = 0
                seconds = [0, 0, 0, 0]
        # crop the images wrt boxes (random crop the margin without boxes), deactivated by default
        if boxes is not None and boxes.sum() != 0:
            imgs, crop_params = custom_img_crop(imgs, boxes, pred=pred)
        else:
            crop_params = torch.tensor([0., 0., 0., 0.])

        im_size = imgs.shape[2:]
        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final, im_size, crop_params, valid, seconds

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def read_feature(path):
    if path.endswith('npz') or path.endswith('npy'):
        return np.load(path)
    if path.endswith('pkl') or path.endswith('pickle'):
        with open(path, 'wb') as f:
            return pickle.load(f)


class DatasetSegmentWrapperSpecialOCv2(TorchDataset):
    def __init__(self):
        pass

########################
########### from github.com/Chuhanxx/helping_hand_for_egocentric_videos/blob/main/data_loader/Egtea.py
import os.path as osp

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def video_loader(root, vid, second, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid)))
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
            vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).to(torch.float32)
        frames = [frame for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


class DatasetWrapperEGTEA(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, transform_crops=None, meta_data_only=False, lab2cname=None):
        self.cfg = cfg
        # data_source include all meta information that should be loaded during init, like labels, mapping and so on
        # dataset wrapper is solely for features / videos / images loading
        self.data_source = data_source
        self.meta_data_only = meta_data_only
        self.transform = transform  # accept list (tuple) as input
        self.toTensor = transforms.ToTensor()
        self.is_train = is_train
        self.label_type = self.cfg.DATASET.LABEL_TYPE
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.use_dino_features = cfg.DATALOADER.USE_DINO_FEATURES
        self.use_dino_features2 = cfg.DATALOADER.USE_DINO_FEATURES2
        self.use_lavila_features = cfg.DATALOADER.LAVILA_FEATURES
        self.use_extracted_DINO_features_only = cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES_ONLY
        self.load_all_features_at_once = cfg.DATALOADER.LOAD_ALL_FEATURES_AT_ONCE
        self.features = None
        self.features_dim = cfg.DATALOADER.FEATURES_DIM
        self.lab2cname = lab2cname
        self.frames_per_seg = cfg.DATALOADER.FRAMES_PER_SEGMENT


        root = os.path.abspath(os.path.expanduser(cfg.DATASET.EGTEA.ROOT))
        self.root = root
        # self.is_trimmed = cfg.DATASET.EGTEA.ROOT.IS_TRIMMED
        anno_dir = cfg.DATASET.EGTEA.ANNOTATIONS_DIR
        self.anno_dir = anno_dir
        # metadata  = cfg.DATASET.EGTEA.METADATA_DIR

        self.num_clips = cfg.DATASET.EGTEA.NUM_CLIPS
        self.clip_length = cfg.DATASET.EGTEA.CLIP_LENGTH
        self.clip_stride = cfg.DATASET.EGTEA.CLIP_STRIDE
        self.sparse_sample = cfg.DATASET.EGTEA.SPARSE_SAMPLE
        self.num_crops = cfg.DATASET.EGTEA.NUM_CROPS

        if self.use_extracted_features and self.load_all_features_at_once:
            root = cfg.DATALOADER.EGTEA.FEATURES_NAME
            split = 'validation' if not self.is_train else 'test'
            if not self.use_extracted_DINO_features_only:
                full_feature_path = os.path.join(root, f'{split}.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load train features')
                    time_start = time.time()
                    self.features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Train features load Elapsed: {elapsed}")
            else:
                self.features = None

            if self.use_dino_features:
                if cfg.DATALOADER.EGTEA.FEATURES_NAME == cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO:
                    self.features_dino = self.features
                    print(f"Train DINO features load Elapsed: {0}")
                else:
                    root = cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO
                    full_feature_path_dino = os.path.join(root, f'{split}.pkl')  # split narration_id
                    with open(full_feature_path_dino, 'rb') as f_dino:
                        # print('load train features')
                        time_start = time.time()
                        self.features_dino = pickle.load(f_dino)
                        elapsed = round(time.time() - time_start)
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(f"Train DINO features load Elapsed: {elapsed}")
            if self.use_dino_features2:
                if cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO == cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO2:
                    self.features_dino2 = self.features_dino
                    print(f"Train DINO features load Elapsed: {0}")
                else:
                    root = cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO2
                    full_feature_path_dino = os.path.join(root, f'{split}.pkl')  # split narration_id
                    with open(full_feature_path_dino, 'rb') as f_dino:
                        # print('load train features')
                        time_start = time.time()
                        self.features_dino2 = pickle.load(f_dino)
                        elapsed = round(time.time() - time_start)
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print(f"Train DINO features load Elapsed: {elapsed}")

        elif self.use_extracted_features:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            # split = 'validation' if not self.is_train else 'test'
            if is_train:
                prefix = 'test_%s.npy'
            else:
                prefix = 'validation_%s.npy'

            # local_path = root + f'segments_npy/{split_file}_{narration_id}.npy'
            root = cfg.DATALOADER.EGTEA.FEATURES_NAME
            self.full_feature_path = os.path.join(root, 'segments_npy', prefix)  # split narration_id
            self.features = None
            if self.use_dino_features:
                root_dino = cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO
                self.full_feature_path_dino = os.path.join(root_dino, 'segments_npy', prefix)
                self.features_dino = None
            if self.use_dino_features2:
                root_dino2 = cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO2
                self.full_feature_path_dino2 = os.path.join(root_dino2, 'segments_npy', prefix)
                self.features_dino2 = None



        self.hand_crops = cfg.DATALOADER.HAND_CROPS
        self.return_full_frame = cfg.DATALOADER.CROPPER.FULL_FRAME




        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize((224, 224)),
            transforms_video.NormalizeVideo(mean=cfg.DATASET.EGTEA.MEAN, # [108.3272985, 116.7460125, 104.09373615000001],
                                            std=cfg.DATASET.EGTEA.STD),   # [68.5005327, 66.6321579, 70.32316305]),
        ])


        self.transform = val_transform



    def __len__(self):
        return len(self.data_source)

    def get_item(self, idx):
        # tmp_only_meta_data = self.meta_data_only
        output = self.__getitem__(idx, meta_data_only=False)
        # self.meta_data_only = tmp_only_meta_data
        return output

    def _uniform_frame_generator(self, start_frame, end_frame):
        if self.frames_per_seg <= -1:  # all frames
            uniform_frames = range(start_frame, end_frame)
        else:
            uniform_frames, step = np.linspace(start_frame, end_frame - 1, self.frames_per_seg, endpoint=False, retstep=True)
            if self.is_train:
                shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                shift = 0
            uniform_frames = [int(i) + shift for i in uniform_frames]
        return uniform_frames


    def __getitem__(self, idx, meta_data_only=None):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]
        # breakpoint()

        output = {
            "domain": 0,
            "impath": '',
            "index": idx,
            "narration_id": item['clip_id'],
            'subclass_label': -1,
            "narration": '',
            "label": item["label"]
        }

        # if self.label_type in ['all', 'all_w_narrations']:
        #     for label_type in ['noun', 'verb', 'action']:
        #         output[f'{label_type}_label'] = item[f'{label_type}_class']
        #
        # else:
        #     output['label'] = item[f'{self.label_type}_class']

        if meta_data_only is None and self.meta_data_only:
            return output



        #################################################
        ################## USING FEATURES ###############
        #################################################

        if self.use_extracted_features:
            start_frame, end_frame = item['start_frame'], item['end_frame']

            output_img_stack = []
            if not self.use_extracted_DINO_features_only:
                if self.frames_per_seg <= -1:  # all frames
                    uniform_frames = range(start_frame, end_frame)
                elif self.frames_per_seg == 0:
                    return output
                else:
                    uniform_frames, step = np.linspace(start_frame, end_frame - 1, self.frames_per_seg, endpoint=False, retstep=True)
                    if self.is_train:
                        shift = np.random.choice(np.arange(int(step))) if int(step) else 0
                    else:
                        shift = 0
                    uniform_frames = [int(i) + shift for i in uniform_frames]

                # try:
                    # if self.features is not None:
                if self.features is not None:
                    features = self.features[item['clip_id']]
                else:
                    features = np.load(self.full_feature_path % item['clip_id'] )
                uniform_frames = [i - start_frame  for i in uniform_frames]
                if self.use_lavila_features:
                    if self.is_train:
                        random_idx = np.random.choice(np.arange(len(features)))
                        output['img'] = torch.tensor(features[random_idx])
                    else:
                        # output['img'] = torch.tensor(features.mean(0))
                        # output['img'] = torch.tensor(features[features.shape[0] // 2])
                        output['img'] = torch.tensor(features[0])
                        # output['img'] = torch.tensor(features[1] if len(features) > 1 else features[0])
                else:
                    output['img'] = torch.tensor(features[uniform_frames])

            if self.use_dino_features:
                if self.features_dino is not None:
                    dino_features = self.features_dino[item['clip_id']]
                else:
                    dino_features = np.load(self.full_feature_path_dino % item['clip_id'])
                uniform_frames_dino = self._uniform_frame_generator(0, len(dino_features))
                output['dino'] = torch.tensor(dino_features[uniform_frames_dino])

            if self.use_dino_features2:
                if self.features_dino2 is not None:
                    dino_features2 = self.features_dino2[item['clip_id']]
                else:
                    dino_features2 = np.load(self.full_feature_path_dino2 % item['clip_id'])

                dino2_subset = torch.tensor(dino_features2[uniform_frames_dino]).unsqueeze(0)
                dino_subset = output['dino'].unsqueeze(0)
                dino_subset = torch.cat([dino_subset, dino2_subset])
                # breakpoint()
                dino_dim = dino_subset.shape[-1]
                dino_subset = dino_subset.transpose(0,1).reshape(-1, dino_dim)
                output['dino'] = dino_subset

            if not self.use_extracted_DINO_features_only:
                return output

        #################################################
        ################## USING VIDEOS #################
        #################################################

        frames, _ = get_raw_item(item,
                                     is_training=self.is_train,
                                     num_clips=self.num_clips,
                                     clip_length=self.clip_length,
                                     clip_stride=self.clip_stride,
                                     sparse_sample=self.sparse_sample)

        # TODO: add crop creation
        # breakpoint()
        if self.transform is not None:
            frames = self.transform(frames)

        frames = frames.transpose(0, 1)
        output['img'] = frames

        return output



    def _get_video_frames(self, video_fp, video_sec, bound_sec, boxes=None, pred=False):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, seconds = self.video_reader(video_fp[0], video_sec[0], end_second=video_sec[1],
                                                  clip_length=self.video_params['num_frames'])
                valid = 1
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0).to(torch.float32)
                valid = 0
                seconds = [0, 0, 0, 0]
        # crop the images wrt boxes (random crop the margin without boxes), deactivated by default
        if boxes is not None and boxes.sum() != 0:
            imgs, crop_params = custom_img_crop(imgs, boxes, pred=pred)
        else:
            crop_params = torch.tensor([0., 0., 0., 0.])

        im_size = imgs.shape[2:]
        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final, im_size, crop_params, valid, seconds

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

