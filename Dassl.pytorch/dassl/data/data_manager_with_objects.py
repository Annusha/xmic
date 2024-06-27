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

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform
# from .utils.dist_utils import get_world_size, get_rank



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
        dataset_wrapper = DatasetWrapper

    # Build data loader
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
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
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
        dataset_wrapper=None
    ):
        # Load dataset
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

        # Build val_loader
        val_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.val,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=None,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
            lab2cname=self._lab2cname
        )

        # Build test_loader
        test_loader = val_loader
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


        if self.use_extracted_features and self.load_all_features_at_once:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            if is_train:
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'train.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load train features')
                    time_start = time.time()
                    self.features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Train features load Elapsed: {elapsed}")

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
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'validation.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load val features')
                    time_start = time.time()
                    self.features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Val features load Elapsed: {elapsed}")

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
                                                           'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO,
                                                           prefix)
                self.features_dino = None
            if self.use_dino_features2:
                self.full_feature_path_dino2 = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL,
                                                            'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME_DINO2,
                                                            prefix)
                self.features_dino2 = None


        # object features right now are the full frames, and then I detect objects there
        # next step would be to include more objects from detic detections
        if self.use_objects_features and not self.detic_crops:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            if is_train:
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.OBJECT_FEATURES_NAME, 'train.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load train features')
                    time_start = time.time()
                    self.object_features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Train object features load Elapsed: {elapsed}")
            else:
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.OBJECT_FEATURES_NAME, 'validation.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load val features')
                    time_start = time.time()
                    self.object_features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Val object features load Elapsed: {elapsed}")


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

        if self.use_objects_features and self.detic_crops:
            split_name = 'train' if is_train else "validation"
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            detic_root = cfg.DATALOADER.DETIC.ROOT
            meta_data_path = f'{root}/detic/{detic_root}_npz/{split_name}_metadata.pkl'
            with open(meta_data_path, 'rb') as f:
                time_start = time.time()
                self.meta_detic_data = pickle.load(f)
                elapsed = round(time.time() - time_start)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"{split_name} meta-data detic load Elapsed: {elapsed}")



        # set up detic mapping to seen subset
        if cfg.DATALOADER.DETIC.CROPS and cfg.DATALOADER.DETIC.SEEN_SUBSET:
            # cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS = 190 if cfg.DATASET.SUBSET == 'seen_nouns' else 300
            assert self.lab2cname is not None
            cname2lab = {v:k for k,v in self.lab2cname.items()}
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            detic_filename = cfg.DATALOADER.DETIC.ROOT
            detic_filename = detic_filename.split('detic_frames_')[-1]
            detic_vocab = os.path.join(root, 'annotations', detic_filename)
            detic_vocab_ordered_list = []
            with open(detic_vocab) as f:
                for line in f:
                    detic_lab, detic_cname = line.strip().split()
                    detic_cname = " ".join(detic_cname.split('_'))
                    detic_vocab_ordered_list.append(detic_cname)
            detic_lab2subset_lab = []
            for class_name in detic_vocab_ordered_list:
                if class_name in cname2lab:
                    detic_lab2subset_lab.append(cname2lab[class_name])
                else:
                    detic_lab2subset_lab.append(-1)

            self.detic_lab2subset_lab = np.array(detic_lab2subset_lab)
            assert self.detic_lab2subset_lab.sum() > 0
        else:
            self.detic_lab2subset_lab = None

    def __len__(self):
        return len(self.data_source)

    def get_item(self, idx):
        # tmp_only_meta_data = self.meta_data_only
        output = self.__getitem__(idx, meta_data_only=False)
        # self.meta_data_only = tmp_only_meta_data
        return output


    def __getitem__(self, idx, meta_data_only=None):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]
        path = item['path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        start_sec = item['start_timestamp']
        end_sec = item['stop_timestamp']

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
                shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                shift = 0
            uniform_frames = [int(i) + shift for i in uniform_frames]


        if self.use_extracted_features:
            # try:
                # if self.features is not None:
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
                    output['img'] = torch.tensor(features[0])
            else:
                output['img'] = torch.tensor(features[uniform_frames])

            if self.use_dino_features:
                if self.features_dino is not None:
                    dino_features = self.features_dino[item['narration_id']]
                else:
                    dino_features = np.load(self.full_feature_path_dino % item['narration_id'])
                output['dino'] = torch.tensor(dino_features[uniform_frames])

            if self.use_dino_features2:
                if self.features_dino2 is not None:
                    dino_features2 = self.features_dino2[item['narration_id']]
                else:
                    dino_features2 = np.load(self.full_feature_path_dino2 % item['narration_id'])
                dino2_subset = torch.tensor(dino_features2[uniform_frames]).unsqueeze(0)
                dino_subset = output['dino'].unsqueeze(0)
                dino_subset = torch.cat([dino_subset, dino2_subset])
                # breakpoint()
                dino_dim = dino_subset.shape[-1]
                dino_subset = dino_subset.transpose(0,1).reshape(-1, dino_dim)
                output['dino'] = dino_subset

            if self.use_objects_features:
                if self.detic_crops:
                    detic_objs = []
                    detic_labels = []

                    detic_obj_features = np.load(item['detic_feature_path'])
                    # the following three lines i can optimize in one-time reading beforehead
                    # detic_meta = np.load(item['detic_meta_path'], allow_pickle=True)
                    # frame_idx2start_end = detic_meta['frame_idx2start_end'].item()  # strange thingy with numpy when dict is safe. it's basically unpacking of it
                    # frame_meta_mat = detic_meta['data']
                    detic_meta = self.meta_detic_data[item['narration_id']]
                    frame_idx2start_end = detic_meta['frame_idx2start_end']
                    # meta data:  object_idx (0), bbox (1,2,3,4) [4 coordinates: x1, y1, x2, y2], object_class_idx (5), object_confidence (6)
                    frame_meta_mat = detic_meta['meta_data']
                    segment_object_labels = frame_meta_mat[:, 5].astype(np.int32)

                    n_obj_per_frame = self.cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME
                    obj_idx_to_collect = []
                    one_hot_mapping = np.eye(self.cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS)
                    collect_object_features = []
                    collect_object_labels = []
                    for enum_frame, frame_idx in enumerate(uniform_frames):
                        start_idx, end_idx = frame_idx2start_end[frame_idx + start_frame]
                        # print(f'{enum_frame} {len(collect_object_labels)}')
                        if start_idx == end_idx:
                            start_idx = min(start_idx, len(detic_obj_features)-1)

                            if self.detic_lab2subset_lab is not None and self.detic_lab2subset_lab[segment_object_labels[start_idx]] > 0:
                                collect_object_labels.extend([self.detic_lab2subset_lab[segment_object_labels[start_idx]]] * n_obj_per_frame)
                                # breakpoint()
                                tmp_feat = detic_obj_features[start_idx].reshape(1,-1)
                                tmp_feat = np.tile(tmp_feat, (1, n_obj_per_frame, 1))
                                collect_object_features.append(tmp_feat)
                                # random_objects = np.array([start_idx] * n_obj_per_frame)
                            else:
                                collect_object_labels.extend([label] * n_obj_per_frame)
                                tmp_feat = output['img'][enum_frame].reshape(1,-1)
                                tmp_feat = np.tile(tmp_feat, (1, n_obj_per_frame, 1))
                                collect_object_features.append(tmp_feat)
                                # collect_object_features.append(tmp_feat)
                                # random_objects = np.array([start_idx] * n_obj_per_frame)
                        else:
                            # breakpoint()
                            valid_idxs = np.where(self.detic_lab2subset_lab[segment_object_labels[start_idx:end_idx]].reshape(-1) > 0)[0]
                            if len(valid_idxs) == 0:
                                collect_object_labels.extend([label] * n_obj_per_frame)
                                tmp_feat = output['img'][enum_frame].reshape(1,-1)
                                tmp_feat = np.tile(tmp_feat, (1, n_obj_per_frame, 1))
                                collect_object_features.append(tmp_feat)
                            else:
                                random_objects = np.random.choice(valid_idxs, n_obj_per_frame)
                                random_objects = [rand_idx + start_idx for rand_idx in random_objects]
                                collect_object_labels.extend(self.detic_lab2subset_lab[segment_object_labels[random_objects]].tolist())
                                tmp_feat = detic_obj_features[random_objects][np.newaxis]
                                collect_object_features.append(tmp_feat)

                        # obj_idx_to_collect.extend(random_objects.tolist())

                    # output['detic_objs'] = detic_obj_features[obj_idx_to_collect].reshape(len(uniform_frames), n_obj_per_frame, -1)
                    # detic_segment_labels = frame_meta_mat[obj_idx_to_collect][:, 5]
                    # output['detic_labels'] =  one_hot_mapping[frame_meta_mat[obj_idx_to_collect][:, 5].reshape(len(uniform_frames), n_obj_per_frame).astype(np.int32)]

                    output['detic_objs'] = np.concatenate(collect_object_features)
                    # detic_segment_labels = np.array(collect_object_labels)
                    output['detic_labels'] =  one_hot_mapping[collect_object_labels].reshape(len(uniform_frames), n_obj_per_frame, -1)

                else:
                    features = self.object_features[item['narration_id']]
                    output['img_obj'] = torch.tensor(features[uniform_frames])


            return output


        # if self.detic_crops:
        #     detic_crops = np.load(item['detic_meta_path'], allow_pickle=True)
        #     frame_idx2start_end = detic_crops['frame_idx2start_end'].item()
        #     data_detic_crops = detic_crops['data']
        #     output['frame_idx2start_end'] = frame_idx2start_end

        #################################################
        ################## USING FRAMES #################
        #################################################

        save_bool = False
        already_saved = False
        for frame_idx in uniform_frames:
            img_i = read_image(path % frame_idx)
            img_full = None

            if self.detic_crops:
                img_i = self.toTensor(img_i)

                start_detic, end_detic = frame_idx2start_end[frame_idx][:]
                cur_frame_objects = data_detic_crops[start_detic:end_detic]
                # object_idx, bbox (4 coordinates: x1, y1, x2, y2), object_class_idx, object_confidence

                output_detic_frame_stack = []
                output_detic_label_stack = []
                for detic_obj in cur_frame_objects:

                    left, top, right, bottom = [int(i) for i in detic_obj[1:5]]
                    if (right - left) > 1 and (bottom - top) > 1:
                        detic_obj_crop = img_i[:, top:bottom, left:right]
                    else:
                        detic_obj_crop = img_i
                    output_detic_label_stack.append(int(detic_obj[5]))

                    if self.transform is not None:
                        detic_obj_crop = self._transform_image(self.transform, detic_obj_crop)
                        output_detic_frame_stack.append(detic_obj_crop.unsqueeze(0))

                if output_detic_frame_stack:
                    output_detic_frame_stack = torch.cat(output_detic_frame_stack, dim=0)
                    output_detic_label_stack = torch.tensor(output_detic_label_stack)

                output['detic_imgs'][frame_idx] = output_detic_frame_stack
                output['detic_labels'][frame_idx] = output_detic_label_stack

            if self.hand_crops:
                # save_bool = False
                # if np.random.random() > 0.5 and not already_saved:
                #     img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}.jpg')
                #     save_bool = True
                #     already_saved = True
                if self.return_full_frame:
                    img_full = img_i.copy()

                if self.mask_background:
                    if isinstance(img_i, Image.Image):
                        img_i = self.toTensor(img_i)
                    black = torch.zeros_like(img_i)
                    # need to fix this one how exactly to do the masking that it looks correctly
                    left, top, right, bottom = item['crop_bb'][frame_idx]
                    # print(f'bbox: {item["crop_bb"][frame_idx]} ;  image size: {black.shape}' )
                    black[:, top:bottom, left:right] = img_i[:, top:bottom, left:right]
                    # trans = transforms.ToPILImage()  # before
                    # img_i = trans(black)  # before
                    img_i = black

                    # img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}_black_crop3.jpg')
                    # print('saved image', self.path_to_save_random_crops + f'/{idx}_{frame_idx}_black_crop.jpg')

                    # img_i = black
                else:
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

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


    def get_detic_item_yield(self, idx):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]
        path = item['path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        label = item[f'{self.label_type}_class']
        feature_path  = item['feature_path']

        output = {
            "label": label,
            "domain": label,
            "impath": '',
            "index": idx,
            "narration_id": item['narration_id'],
            'subclass_label': item['subclass_label'] if 'subclass_label' in item else None,
            "n_frames": end_frame - start_frame
        }

        # TODO: check what is faster to collect images and then apply transform to all of them at the same time,
        # or apply transform to each image separately and then concatenate them
        # if self.hand_crops:
            # video_detections = load_detections(item['dection_path'])
        output_img_stack = []
        if self.frames_per_seg == -1: # all frames
            uniform_frames = range(start_frame, end_frame+1)
        elif self.frames_per_seg == 0:
            return output
        else:
            uniform_frames, step = np.linspace(start_frame, end_frame, self.frames_per_seg, endpoint=False, retstep=True)
            if self.is_train:
                shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                shift = 0
            uniform_frames = [int(i) + shift for i in uniform_frames]

        if self.detic_crops:
            output['detic_imgs'] = {}
            output['detic_labels'] = {}

        if self.detic_crops:
            detic_crops = np.load(item['detic_meta_path'], allow_pickle=True)
            frame_idx2start_end = detic_crops['frame_idx2start_end'].item()
            data_detic_crops = detic_crops['data']
            output['frame_idx2start_end'] = frame_idx2start_end

        for frame_idx in uniform_frames:
            img_i = read_image(path % frame_idx)
            img_full = None

            if self.detic_crops:
                img_i = self.toTensor(img_i)

                start_detic, end_detic = frame_idx2start_end[frame_idx][:]
                cur_frame_objects = data_detic_crops[start_detic:end_detic]
                # object_idx, bbox (4 coordinates: x1, y1, x2, y2), object_class_idx, object_confidence

                output_detic_frame_stack = []
                output_detic_label_stack = []
                for detic_obj in cur_frame_objects:

                    left, top, right, bottom = [int(i) for i in detic_obj[1:5]]
                    right = right+1
                    bottom = bottom+1
                    if (right - left) > 1 and (bottom - top) > 1:
                        detic_obj_crop = img_i[:, top:bottom, left:right]
                    else:
                        detic_obj_crop = img_i
                        print(f'{item["narration_id"]}  {detic_obj[0]} | {left} {top} {right} {bottom} | {detic_obj[-2]} {detic_obj[-1]}')
                    output_detic_label_stack.append(int(detic_obj[5]))

                    if self.transform is not None:
                        detic_obj_crop = self._transform_image(self.transform, detic_obj_crop)
                        output_detic_frame_stack.append(detic_obj_crop.unsqueeze(0))

                if output_detic_frame_stack:
                    output_detic_frame_stack = torch.cat(output_detic_frame_stack, dim=0)
                    output_detic_label_stack = torch.tensor(output_detic_label_stack)

                # output['detic_imgs'][frame_idx] = output_detic_frame_stack
                # output['detic_labels'][frame_idx] = output_detic_label_stack

                yield frame_idx, output_detic_frame_stack, frame_idx2start_end



def read_feature(path):
    if path.endswith('npz') or path.endswith('npy'):
        return np.load(path)
    if path.endswith('pkl') or path.endswith('pickle'):
        with open(path, 'wb') as f:
            return pickle.load(f)



class DatasetSegmentWrapperSpecialOCv2(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, transform_crops=None, meta_data_only=False):
        self.cfg = cfg
        self.data_source = data_source
        self.meta_data_only = meta_data_only
        self.transform = transform  # accept list (tuple) as input
        self.toTensor = transforms.ToTensor()
        self.is_train = is_train
        self.label_type = self.cfg.DATASET.LABEL_TYPE
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.features = None
        self.features_dim = cfg.DATALOADER.FEATURES_DIM
        if self.use_extracted_features:
            root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
            if is_train:
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'train.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load train features')
                    time_start = time.time()
                    self.features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Train features load Elapsed: {elapsed}")
            else:
                full_feature_path = os.path.join(root, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'validation.pkl')  # split narration_id
                with open(full_feature_path, 'rb') as f:
                    # print('load val features')
                    time_start = time.time()
                    self.features = pickle.load(f)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"Val features load Elapsed: {elapsed}")


        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.frames_per_seg = cfg.DATALOADER.FRAMES_PER_SEGMENT # TODO: add to config
        crop_file_name = f'hand_thd{cfg.DATALOADER.CROPPER.HAND_THS}_obj_thd{cfg.DATALOADER.CROPPER.OBJ_THS}{"_ONLY_inter_obj" if cfg.DATALOADER.CROPPER.ONLY_INTERACTED_OBJ else ""}{"_with_HANDS" if cfg.DATALOADER.CROPPER.WITH_HANDS else ""}{cfg.DATALOADER.CROPPER.TAG}'
        self.path_to_save_random_crops = os.path.join(os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT)), 'annotations', 'hand_crops', 'croped_random_frames', f'{crop_file_name}')
        os.makedirs(self.path_to_save_random_crops, exist_ok=True)
        # self.path_to_save_random_crops = f'{cfg.DATASET.DETECTION_ROOT}/croped_random_frames/{crop_file_name}/'

        self.hand_crops = cfg.DATALOADER.HAND_CROPS
        self.return_full_frame = cfg.DATALOADER.CROPPER.FULL_FRAME
        self.mask_background = cfg.DATALOADER.CROPPER.BLACK_CROP

        self.detic_crops = cfg.DATALOADER.DETIC.CROPS

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

    def get_item(self, idx):
        # tmp_only_meta_data = self.meta_data_only
        output = self.__getitem__(idx, meta_data_only=False)
        # self.meta_data_only = tmp_only_meta_data
        return output


    def __getitem__(self, idx, meta_data_only=None):
        # print('Get item', idx, flush=True)
        item = self.data_source[idx]
        path = item['path']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        label = item[f'{self.label_type}_class']
        feature_path  = item['feature_path']

        output = {
            "label": label,
            "domain": label,
            "impath": '',
            "index": idx,
            "narration_id": item['narration_id'],
            'subclass_label': item['subclass_label'] if 'subclass_label' in item else -1
        }

        if meta_data_only is None and self.meta_data_only:
            return output

        # TODO: check what is faster to collect images and then apply transform to all of them at the same time,
        # or apply transform to each image separately and then concatenate them
        # if self.hand_crops:
            # video_detections = load_detections(item['dection_path'])
        output_img_stack = []
        if self.frames_per_seg == -1: # all frames
            uniform_frames = range(start_frame, end_frame)
        elif self.frames_per_seg == 0:
            return output
        else:
            uniform_frames, step = np.linspace(start_frame, end_frame-1, self.frames_per_seg, endpoint=False, retstep=True)
            if self.is_train:
                shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                shift = 0
            uniform_frames = [int(i) + shift for i in uniform_frames]

        if self.detic_crops:
            output['detic_imgs'] = {}
            output['detic_labels'] = {}

        if self.use_extracted_features:
            # try:
                # if self.features is not None:
            features = self.features[item['narration_id']]
            # else:
            #     features = read_feature(feature_path)
            uniform_frames = [i - start_frame for i in uniform_frames]
            output['img'] = torch.tensor(features[uniform_frames])
            # except Exception:
            #     output['img'] = torch.zeros(self.frames_per_seg, self.features_dim)

            return output


        if self.detic_crops:
            detic_crops = np.load(item['detic_objects_path'], allow_pickle=True)
            frame_idx2start_end = detic_crops['frame_idx2start_end'].item()
            data_detic_crops = detic_crops['data']
            output['frame_idx2start_end'] = frame_idx2start_end

        for frame_idx in uniform_frames:
            img_i = read_image(path % frame_idx)
            img_full = None

            if self.detic_crops:
                img_i = self.toTensor(img_i)

                start_detic, end_detic = frame_idx2start_end[frame_idx][:]
                cur_frame_objects = data_detic_crops[start_detic:end_detic]
                # object_idx, bbox (4 coordinates: x1, y1, x2, y2), object_class_idx, object_confidence

                output_detic_frame_stack = []
                output_detic_label_stack = []
                for detic_obj in cur_frame_objects:

                    left, top, right, bottom = [int(i) for i in detic_obj[1:5]]
                    detic_obj_crop = img_i[:, top:bottom, left:right]
                    output_detic_label_stack.append(int(detic_obj[5]))

                    if self.transform is not None:
                        detic_obj_crop = self._transform_image(self.transform, detic_obj_crop)
                        output_detic_frame_stack.append(detic_obj_crop.unsqueeze(0))

                if output_detic_frame_stack:
                    output_detic_frame_stack = torch.cat(output_detic_frame_stack, dim=0)
                    output_detic_label_stack = torch.tensor(output_detic_label_stack)

                output['detic_imgs'][frame_idx] = output_detic_frame_stack
                output['detic_labels'][frame_idx] = output_detic_label_stack

            if self.hand_crops:
                # save_bool = False
                # if np.random.random() > 0.5:
                #     img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}.jpg')
                #     save_bool = True
                if self.return_full_frame:
                    img_full = img_i.copy()

                if self.mask_background:
                    if isinstance(img_i, Image.Image):
                        img_i = self.toTensor(img_i)
                    black = torch.zeros_like(img_i)
                    # region to black out
                    left, top, right, bottom = item['crop_bb_v2'][frame_idx]
                    if (bottom - top) < black.shape[1] * 0.7 and (right - left) < black.shape[2] * 0.95:
                    # print(f'bbox: {item["crop_bb"][frame_idx]} ;  image size: {black.shape}' )
                        img_i[:, top:bottom, left:right] = black[:, top:bottom, left:right]
                    # actual region to crop
                    left, top, right, bottom = item['crop_bb'][frame_idx]
                    if (right - left) > 1 and (bottom - top) > 1:
                        img_i = img_i[:, top:bottom, left:right]

                else:
                    left, top, right, bottom = item['crop_bb'][frame_idx]
                    if (right - left) > 1 and (bottom - top) > 1:
                        if isinstance(img_i, Image.Image):
                            img_i = img_i.crop(item['crop_bb'][frame_idx])
                        else:
                            img_i = img_i[:, top:bottom, left:right]
                # if save_bool:
                    # img_i.save(self.path_to_save_random_crops + f'/{idx}_{frame_idx}_crop.jpg')
            #     img_i = self.cropper.render_crops(img_i, video_detections[frame_idx])
                # print(f'Get cropped frame {frame_idx}')


            if self.transform is not None:
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
        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
