from __future__ import annotations
import gc

import json
import logging
import os
import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pickle
import datetime
import time
import torchvision.transforms as T
from torchvision.transforms import Resize, ToTensor

from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsampleRepeated,
)
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)

from torchvision.transforms import (
    Compose,
    Lambda,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC



# [
#             ShortSideScale(cfg.DATA.TRAIN_JITTER_SCALES[0]),
#             CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
#         ]

class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
        save_root: str='',
        split: str = 'train',
        seed: int = 0,
        cfg: dict = None,
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self.path_handler = VideoPathHandler()
        self._save_root = save_root
        self._split = split
        self._seed = seed
        self.cfg = cfg

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clips = None
        self._next_clip_start_time = 0.0

        self._transform_inner = Compose([Lambda(lambda x: x / 255.0),
                                         Resize(224, interpolation=BICUBIC),
                                         CenterCrop(224),
                                         Normalize(cfg.DATA.MEAN, cfg.DATA.STD)])
                     # Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                     # ShortSideScale(cfg.DATA.TRAIN_JITTER_SCALES[0]),
                     # CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),])

        if self.cfg is not None and self.cfg.DATALOADER.HAND_CROPS:
            root = self.cfg.DATA.PATH_TO_DATA_DIR
            print(f'Loaded hand crops: {self.cfg.DATALOADER.HAND_CROPS_NAME}_{split}.pkl')
            crop_path = f'{root}/hand_crops/{self.cfg.DATALOADER.HAND_CROPS_NAME}_{split}.pkl' # pickle file
            self.path_to_save_random_crops = '/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/hand_obj_detections/visualizations/'
            with open(crop_path, 'rb') as f:
                self.crops = pickle.load(f)

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))


        for _ in range(len(self._labeled_videos)):
            video_index = next(self._video_sampler_iter)
            # video_path, info_dict = self._labeled_videos[video_index]
            # idxs.append(info_dict["action_idx"])
            # breakpoint()
            # if len(idxs) == 20:
            #     rank = int(os.environ["LOCAL_RANK"])
            #     print(rank, idxs)
            # continue

            for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    # check here if the video already extracted or not
                    # if the features already exist, move to the next video action_idx
                    video_name = f'{info_dict["clip_uid"]}_{info_dict["action_idx"]}'
                    full_feature_path = os.path.join(self._save_root, f'{self._split}_{info_dict["clip_uid"]}_{info_dict["action_idx"]}.npy')
                    if os.path.exists(full_feature_path):
                        break
                    # print('extract features: ', video_path)
                    video = self.path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                except Exception as e:
                    print(
                        "Failed to load video with error: {}; trial {}".format(e, i_try)
                    )
                    continue

                if self.cfg.DATALOADER.DECODE_VIDEO_AT_ONCE:
                    start_sec = info_dict['clip_start_sec']
                    end_sec = info_dict['clip_end_sec'] + 0.9
                    clip_frames = video.get_clip(start_sec, end_sec)
                    frames = clip_frames['video']
                    clip_index = 0
                    aug_index = 0
                    audio_samples = None
                    # print('SHAPE', frames.shape)
                else:

                    clips = self._clip_sampler(
                        self._next_clip_start_time, video.duration, info_dict
                    )

                    if not isinstance(clips, list):
                        clips = [clips]

                    # print('len clips', len(clips))

                    decoded_clips = []
                    video_is_null = False
                    for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
                        # print(info_dict)
                        # print(clip_start, clip_end)
                        clip = video.get_clip(clip_start, clip_end)
                        video_is_null = clip is None or clip["video"] is None
                        if video_is_null:
                            break
                        decoded_clips.append(clip)

                    self._next_clip_start_time = clip_end

                    if is_last_clip or video_is_null:
                        # Close the loaded encoded video and reset the last sampled clip time ready
                        # to sample a new video on the next iteration.
                        video.close()
                        self._next_clip_start_time = 0.0

                        # Force garbage collection to release video container immediately
                        # otherwise memory can spike.
                        gc.collect()

                        if video_is_null:
                            print(
                                "Failed to load clip {}; trial {}".format(video.name, i_try)
                            )
                            continue


                    if len(decoded_clips) == 1:
                        # this is what happen during training
                        frames = decoded_clips[0]["video"]
                        audio_samples = decoded_clips[0]["audio"]
                    else:
                        # this is what happen during testing
                        # clip_frames = [
                        #     uniform_temporal_subsample(x["video"], num_samples=64)
                        #     for x in decoded_clips
                        # ]
                        clip_frames = [
                            x["video"].squeeze()
                            for x in decoded_clips
                        ]
                        # stack along temporal dimension
                        # CxTxHxW
                        # print(clip_frames[0].shape)
                        frames = torch.cat(clip_frames, dim=1)

                        clip_audio = [x["audio"] for x in decoded_clips]
                        audio_samples = None
                        if None not in clip_audio:
                            audio_samples = torch.stack(clip_audio, dim=0)

                # breakpoint()
                if self.cfg.DATALOADER.HAND_CROPS:
                    crop_bb = self.crops[video_name]
                    len_crops = len(crop_bb)
                    min_frame_id = min(list(crop_bb.keys()))
                    try:
                        # assert len_crops <= frames.shape[1] + 1 and len_crops >= frames.shape[1]
                        assert (frames.shape[1] + 1) >= len_crops
                    except AssertionError:
                        print(frames.shape, len_crops)
                        # print(video.rate)
                        breakpoint()
                    new_frames = []
                    for frame_id in range(min(len_crops, frames.shape[1])):
                        cur_frame = frames[:, frame_id]
                        cur_crop = crop_bb[frame_id + min_frame_id]
                        left, top, right, bottom = cur_crop
                        if left == 0 and top == 0 and right == 0 and bottom == 0:
                            pass
                        else:
                            # breakpoint()
                            # if torch.rand(1).item() > 0.98:
                            #     transform2PIL = T.ToPILImage()
                            #     img_i = transform2PIL(cur_frame)
                            #     img_i.save(self.path_to_save_random_crops + f'/7.10.{video_name}_{frame_id}_full.jpg')
                            # else:
                            #     transform2PIL = None

                            cur_frame = cur_frame[:, top:bottom, left:right]
                            # if transform2PIL is not None:
                            #     img_i = self._transform_inner(cur_frame.unsqueeze(1))
                            #     img_i = transform2PIL(img_i.squeeze())
                            #     img_i.save(self.path_to_save_random_crops + f'/7.10.{video_name}_{frame_id}_crop.jpg')


                        cur_frame = self._transform_inner(cur_frame.unsqueeze(1))
                        new_frames.append(cur_frame)
                        # ToDO: add visualization that crops are extracted correctly
                    # breakpoint()
                    frames = torch.cat(new_frames, axis=1)
                    # print(frames.shape)


                sample_dict = {
                    "video": frames,
                    "video_name": video.name,
                    "video_index": video_index,
                    "clip_index": clip_index,
                    "aug_index": aug_index,
                    "full_feature_path": full_feature_path,
                    **info_dict,
                    **({"audio": audio_samples} if audio_samples is not None else {}),
                }
                if self._transform is not None:
                    sample_dict = self._transform(sample_dict)

                return sample_dict
            else:
                raise RuntimeError(
                    f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
                )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        try:
            rank = int(os.environ["LOCAL_RANK"])
        except Exception:
            rank = 0
        if self._video_random_generator is not None and worker_info is not None:
            pass
            base_seed = worker_info.seed - worker_info.id + self._seed + rank
            # # print(f'{rank} SEED', base_seed)
            self._video_random_generator.manual_seed(base_seed)
        else:
            base_seed = self._seed + rank
            # print(f'{rank} SEED_no_seed1', base_seed)
            self._video_random_generator.manual_seed(base_seed)

        return self

def parse_class(x):
    if '(' not in x:
        x = " ".join(x.split('_'))
        return [x]
    x = x.split('(')
    word = x[0]
    word = " ".join(word.split('_'))
    rest = x[1][:-1]
    rest = rest.split(',')
    rest = [" ".join(i.split('_')) for i in rest]
    return [word] + rest

def clip_recognition_dataset_feature_extraction(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    classnames_and_mapping: bool = False,
    save_root='',
    split='train',
    seed=0,
    cfg=None,
):

    assert os.path.exists(data_path), 'Please run data/parse_ego4d_json.py first. Will change this later'

    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                annotations = json.load(f)
                annotations = annotations['clips']
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict). For recognition, the annotation_dict contains
        # the verb and noun label, and the annotation boundaries.
        untrimmed_clip_annotations = []
        # container_nouns = set()
        # container_verbs = set()


        mapping_verb_l2c = {}
        mapping_verb_c2l = {}
        train_class_counts_nouns = defaultdict(int)
        with open(f'{os.path.dirname(data_path)}/mappping_verb.txt', 'r') as mapping_f:
            for line in mapping_f:
                line = line.strip().split()
                class_idx = int(line[0])
                mapping_verb_l2c[class_idx] = line[1]
                mapping_verb_c2l[line[1]] = class_idx

        mapping_noun_l2c = {}
        mapping_noun_c2l = {}
        train_class_counts_verbs = defaultdict(int)
        with open(f'{os.path.dirname(data_path)}/mappping_noun.txt', 'r') as mapping_f:
            for line in mapping_f:
                line = line.strip().split()
                class_idx = int(line[0])
                mapping_noun_l2c[class_idx] = line[1]
                mapping_noun_c2l[line[1]] = class_idx



        for entry in tqdm(annotations, "Populating Dataset", total=len(annotations)):
            # print('Entry', entry)
            # print(video_path_prefix)
            # print(entry["clip_uid"])
            # container_nouns.add((entry['noun_label'], entry['noun'].split('_')[0]))
            # container_verbs.add((entry['verb_label'], entry['verb'].split('_')[0]))

            # noun = entry['noun'].split('_')[0]
            # verb = entry['verb'].split('_')[0]
            noun = entry['noun']
            verb = entry['verb']
            train_class_counts_nouns[mapping_noun_c2l[noun]] += 1
            train_class_counts_verbs[mapping_verb_c2l[verb]] += 1


            untrimmed_clip_annotations.append(
                (
                    os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
                    {
                        "clip_start_sec": entry['action_clip_start_sec'],
                        "clip_end_sec": entry['action_clip_end_sec'],
                        # "noun_label": entry['noun_label'],
                        # "verb_label": entry['verb_label'],
                        "noun_label": mapping_noun_c2l[noun],
                        "verb_label": mapping_verb_c2l[verb],
                        "action_idx": entry['action_idx'],
                        "clip_uid": entry["clip_uid"]
                    },
                )
            )
        if classnames_and_mapping:
            # mapping_nouns = {label: classname for label, classname in container_nouns}
            # mapping_verbs = {label: classname for label, classname in container_verbs}

            ## FIX
            def _fix_classes(mapping_nouns_label2classname):
                for k,v in mapping_nouns_label2classname.items():
                    if v.startswith('nut_'):
                        if 'food' in v:
                            mapping_nouns_label2classname[k] = 'nut_food'
                        elif 'tool' in v:
                            mapping_nouns_label2classname[k] = 'nut_tool'
                    if v.startswith('chip_'):
                        if 'food' not in v:
                            if "wood\'" in v:
                                mapping_nouns_label2classname[k] = 'chip_wood'
                            else:
                                mapping_nouns_label2classname[k] = 'chip_metal'
                    if v.startswith('bat_'):
                        if 'tool' in v:
                            mapping_nouns_label2classname[k] = 'bat_tool'
                        elif 'sports' in v:
                            mapping_nouns_label2classname[k] = 'bat_sports'

                    v = mapping_nouns_label2classname[k]
                    mapping_nouns_label2classname[k] = v.split('(')[0]

                return mapping_nouns_label2classname

            mapping_noun_l2c = _fix_classes(mapping_noun_l2c)

            labels_nouns = list(mapping_noun_l2c.keys())
            print('LEN LABELS NOUNS', len(labels_nouns), flush=True)
            labels_nouns.sort()
            # mapping_noun_l2c = {k:v.split('(')[0] for k,v in mapping_noun_l2c.items()}

            classnames_nouns = [mapping_noun_l2c[label] for label in labels_nouns]
            print('CLASSNAMES NOUNS', classnames_nouns, flush=True)
            print('CLASSNAMES NOUNS', len(classnames_nouns), flush=True)

            labels_verbs = list(mapping_verb_l2c.keys())
            print('LEN LABELS VERBS', len(labels_verbs), flush=True)
            labels_verbs.sort()
            # mapping_verb_l2c = {k: v.split('(')[0] for k, v in mapping_verb_l2c.items()}
            mapping_verb_l2c = _fix_classes(mapping_verb_l2c)
            classnames_verbs = [mapping_verb_l2c[label] for label in labels_verbs]
            print('CLASSNAMES VERBS', classnames_verbs, flush=True)
            print('CLASSNAMES VERBS', len(classnames_verbs), flush=True)
            addiional_output = {'mapping_nouns': mapping_noun_l2c, 'classnames_nouns': classnames_nouns, 'train_class_counts_noun': train_class_counts_nouns,
                                'mapping_verbs': mapping_verb_l2c, 'classnames_verbs': classnames_verbs, 'train_class_counts_verb': train_class_counts_verbs}

    else:
        raise FileNotFoundError(f"{data_path} not found.")

    print(clip_sampler)
    print(len(untrimmed_clip_annotations))
    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        UniformUntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        save_root=save_root,
        split=split,
        seed=seed,
        cfg=cfg,
    )
    if classnames_and_mapping:
        return dataset, addiional_output
    return dataset

############################################################
####################  video_transformer ####################
############################################################

from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsampleRepeated,
)
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)


"""
    video transform method to normalize, crop, scale, etc.
"""


def random_scale_crop_flip(mode: str, cfg):
    return (
        [
            RandomShortSideScale(
                min_size=cfg.DATA.TRAIN_JITTER_SCALES[0],
                max_size=cfg.DATA.TRAIN_JITTER_SCALES[1],
            ),
            RandomCrop(cfg.DATA.TRAIN_CROP_SIZE),
            RandomHorizontalFlip(p=cfg.DATA.RANDOM_FLIP),
        ]
        if mode == "train"
        else [
            ShortSideScale(cfg.DATA.TRAIN_JITTER_SCALES[0]),
            CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
        ]
    )


def uniform_temporal_subsample_repeated(cfg):
    return UniformTemporalSubsampleRepeated(
        ((1,))
    )

class UntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_end_boundary - clip_start_boundary

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
        return ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        )

class UniformUntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_end_boundary - clip_start_boundary + 0.9
        clip_info_init = clip_info

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        collect = []
        clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info_init)
        last_clip_time = clip_info.clip_end_sec
        collect.append( ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        ) )
        while not collect[-1].is_last_clip:
            clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info_init)
            last_clip_time = clip_info.clip_end_sec
            collect.append(ClipInfo(
                clip_info.clip_start_sec + clip_start_boundary,
                clip_info.clip_end_sec + clip_start_boundary,
                clip_info.clip_index,
                clip_info.aug_index,
                clip_info.is_last_clip,
            ))
        return collect