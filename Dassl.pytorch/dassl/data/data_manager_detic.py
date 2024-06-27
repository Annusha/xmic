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
    dataset_wrapper=None
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
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
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
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

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
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

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
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
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


class DatasetSegmentWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, transform_crops=None, meta_data_only=False):
        self.cfg = cfg
        self.data_source = data_source
        self.meta_data_only = meta_data_only
        self.transform = transform  # accept list (tuple) as input
        self.toTensor = transforms.ToTensor()
        self.is_train = is_train
        self.label_type = self.cfg.DATASET.LABEL_TYPE
        self.use_extracted_features = cfg.DATALOADER.USE_EXTRACTED_FEATURES
        self.use_objects_features = cfg.DATALOADER.USE_OBJECTS_FEATURES
        self.features = None
        self.object_features = None
        self.features_dim = cfg.DATALOADER.FEATURES_DIM
        self.detic_crops = cfg.DATALOADER.DETIC.CROPS


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

        # set up detic mapping to seen subset
        if cfg.DATALOADER.DETIC.CROPS and cfg.DATALOADER.DETIC.SEEN_SUBSET:
            cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS = 190 if self.subset == 'seen_nouns' else 300

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
            'subclass_label': item['subclass_label'] if 'subclass_label' in item else None
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


        if self.use_extracted_features:
            # try:
                # if self.features is not None:
            features = self.features[item['narration_id']]
            # else:
            #     features = read_feature(feature_path)
            uniform_frames = [i - start_frame for i in uniform_frames]
            output['img'] = torch.tensor(features[uniform_frames])
            if self.use_objects_features:
                if self.detic_crops:
                    detic_objs = []
                    detic_labels = []

                    detic_obj_features = np.load(item['detic_feature_path'])
                    # the following three lines i can optimize in one-time reading beforehead
                    detic_meta = np.load(item['detic_meta_path'], allow_pickle=True)
                    frame_idx2start_end = detic_meta['frame_idx2start_end'].item()  # strange thingy with numpy when dict is safe. it's basically unpacking of it
                    # meta data:  object_idx (0), bbox (1,2,3,4) [4 coordinates: x1, y1, x2, y2], object_class_idx (5), object_confidence (6)
                    frame_meta_mat = detic_meta['data']
                    n_obj_per_frame = self.cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME
                    obj_idx_to_collect = []
                    one_hot_mapping = np.eye(self.cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS)
                    collect_object_features = []
                    for frame_idx in uniform_frames:
                        start_idx, end_idx = frame_idx2start_end[frame_idx + start_frame]
                        if start_idx == end_idx:
                            start_idx = min(start_idx, len(detic_obj_features)-1)
                            random_objects = np.array([start_idx] * n_obj_per_frame)
                        else:
                            random_objects = np.random.choice(range(start_idx, end_idx), n_obj_per_frame)
                        obj_idx_to_collect.extend(random_objects.tolist())

                    output['detic_objs'] = detic_obj_features[obj_idx_to_collect].reshape(len(uniform_frames), n_obj_per_frame, -1)
                    detic_segment_labels = frame_meta_mat[obj_idx_to_collect][:, 5]
                    output['detic_labels'] =  one_hot_mapping[frame_meta_mat[obj_idx_to_collect][:, 5].reshape(len(uniform_frames), n_obj_per_frame).astype(np.int32)]

                else:
                    features = self.object_features[item['narration_id']]
                    output['img_obj'] = torch.tensor(features[uniform_frames])


            return output


        # if self.detic_crops:
        #     detic_crops = np.load(item['detic_objects_path'], allow_pickle=True)
        #     frame_idx2start_end = detic_crops['frame_idx2start_end'].item()
        #     data_detic_crops = detic_crops['data']
        #     output['frame_idx2start_end'] = frame_idx2start_end

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
            'subclass_label': item['subclass_label'] if 'subclass_label' in item else None
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
