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
import pandas as pd

from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler


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

        video_index = next(self._video_sampler_iter)

        return self.get_item(video_index)

    def get_item(self, video_index):

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            try:
                video_path, info_dict = self._labeled_videos[video_index]
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

            # narration = f'{self._split}_{info_dict["clip_uid"]}_{info_dict["action_idx"]}'
            clips = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if not isinstance(clips, list):
                clips = [clips]

            decoded_clips = []
            video_is_null = False
            for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
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
                clip_frames = [
                    uniform_temporal_subsample(x["video"], num_samples=64)
                    for x in decoded_clips
                ]
                frames = torch.stack(clip_frames, dim=0)

                clip_audio = [x["audio"] for x in decoded_clips]
                audio_samples = None
                if None not in clip_audio:
                    audio_samples = torch.stack(clip_audio, dim=0)

            sample_dict = {
                "video": frames,
                # "narration_id": narration,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
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
        # print('Worker info', worker_info)
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


class LabeledFeaturesDataset(torch.utils.data.IterableDataset):

    def __init__(
            self,
            labeled_video_paths: List[Tuple[str, Optional[dict]]],
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
            feature_root: str = '',
            split: str = 'train',
            n_frames: int = 8,
            cfg=None,
    ) -> None:

        self._labeled_videos = labeled_video_paths
        self._feature_root = feature_root
        self._dino_feature_root = cfg.DATA.PATH_PREFIX_DINO
        self._dino_feature_root2 = cfg.DATA.PATH_PREFIX_DINO2
        self._split = split
        self.n_frames = n_frames
        self.cfg = cfg
        self.use_dino_features = cfg.DATALOADER.USE_DINO_FEATURES
        self.use_dino_features2 = cfg.DATALOADER.USE_DINO_FEATURES2
        self.use_lavila_features = cfg.DATALOADER.LAVILA_FEATURES
        self.use_extracted_DINO_features_only = cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES_ONLY
        self.load_all_features_at_once = cfg.DATALOADER.LOAD_ALL_FEATURES_AT_ONCE
        self.clip_length = cfg.DATA.EGO4D_CLIPS_LENGTH

        if self.cfg.TEST.EVAL_ONLY and split == 'train':
            self.features = None
        # elif cfg.TEST.CROSS_DATASET.EVAL and 'Ego4D' in self.cfg.TEST.CROSS_DATASET.DATASET_NAME and split == 'train':
        #     self.features = None
        elif self.load_all_features_at_once:
            if not self.use_extracted_DINO_features_only:
                full_feature_path = os.path.join(self._feature_root, f'{self._split}.pkl')
                time_start = time.time()
                with open(full_feature_path, 'rb') as f:
                    self.features = pickle.load(f)
                elapsed = round(time.time() - time_start)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"{split} features load Elapsed: {elapsed}")
            else:
                self.features = None

            if self.use_dino_features:
                if self._dino_feature_root == self._feature_root:
                    self.features_dino = self.features
                    print(f"{split} DINO features load Elapsed: {0}")
                else:
                    full_feature_path_dino = os.path.join(self._dino_feature_root, f'{self._split}.pkl')
                    time_start = time.time()
                    with open(full_feature_path_dino, 'rb') as f_dino:
                        self.features_dino = pickle.load(f_dino)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"{split} DINO features load Elapsed: {elapsed}")

            if self.use_dino_features2:
                if self._dino_feature_root2 == self._dino_feature_root:
                    self.features_dino2 = self.features_dino
                    print(f"{split} DINO features load Elapsed: {0}")
                else:
                    full_feature_path_dino = os.path.join(self._dino_feature_root2, f'{self._split}.pkl')
                    time_start = time.time()
                    with open(full_feature_path_dino, 'rb') as f_dino:
                        self.features_dino2 = pickle.load(f_dino)
                    elapsed = round(time.time() - time_start)
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print(f"{split} DINO features load Elapsed: {elapsed}")

        else:
            print('No load features,  load on the fly')
            self.features = None
            self.full_feature_path = os.path.join(self._feature_root, 'segments', '%s.npy')
            if self.use_dino_features:
                self.features_dino = None
                self.full_feature_path_dino = os.path.join(self._dino_feature_root, 'segments', '%s.npy')
            if self.use_dino_features2:
                self.features_dino2 = None
                self.full_feature_path_dino2 = os.path.join(self._dino_feature_root2, 'segments', '%s.npy')


        # narration_id = f'{split_file}_{row_segment["clip_uid"]}_{row_segment["action_idx"]}'
        # local_path = root + f'segments/{narration_id}.npy'


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

    @property
    def video_sampler(self):
        return self._video_sampler

    @property
    def num_videos(self):
        return len(self.video_sampler)

    def start_end_clip(self, features_len):
        if self.clip_length == 8:
            clip_start = 0
            clip_end = features_len
        else:
            frames_per_sec = features_len // 8
            mid_frame = features_len // 2
            half_needed = self.clip_length // 2
            clip_start = max(0, mid_frame - frames_per_sec * half_needed)
            clip_end = min(mid_frame + frames_per_sec * half_needed, features_len)
        return clip_start, clip_end


    def __next__(self) -> dict:
        worker_info = torch.utils.data.get_worker_info()
        # print(f"{worker_info=}")

        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        video_index = next(self._video_sampler_iter)
        return self.get_item(video_index)

    def get_item(self, video_index):

        video_path, info_dict = self._labeled_videos[video_index]
        # full_feature_path = os.path.join(self._feature_root,
        #                                  f'{self._split}_{info_dict["clip_uid"]}_{info_dict["action_idx"]}.npy')
        #
        narration = f'{self._split}_{info_dict["clip_uid"]}_{info_dict["action_idx"]}'
        frames_per_seg = self.cfg.DATALOADER.FRAMES_PER_SEGMENT
        if not self.use_extracted_DINO_features_only:
            if self.features is not None:
                features = self.features[f'{self._split}_{info_dict["clip_uid"]}_{info_dict["action_idx"]}']
            else:
                features = np.load(self.full_feature_path % narration)


            clip_start, clip_end = self.start_end_clip(len(features))

            uniform_frames, step = np.linspace(clip_start, clip_end, frames_per_seg, endpoint=False, retstep=True)
            if self._split == 'train':
                if int(step):
                    uniform_frames = [int(i) + np.random.choice(np.arange(int(step))) for i in uniform_frames]
                else:
                    uniform_frames = [int(i) for i in uniform_frames]
                # shift = np.random.choice(np.arange(int(step))) if int(step) else 0
            else:
                # shift = 0
                uniform_frames = [int(i + step / 2) for i in uniform_frames]
            # uniform_frames = [int(i) + shift for i in uniform_frames]
            if self.use_lavila_features:
                if self._split == 'train':
                    random_idx = np.random.choice(np.arange(len(features)))
                else:
                    random_idx = 0
                out_features = torch.tensor(features[random_idx])
            else:
                # breakpoint()
                out_features = torch.tensor(features[uniform_frames])

        else:
            out_features = None
            features = None

        sample_dict = {
            "img": out_features,
            "narration_id": narration,
            "video_index": video_index,
            **info_dict,
        }

        if self.cfg.DATASET.LABEL_TYPE == 'noun':
            label = info_dict['noun_label']
        if self.cfg.DATASET.LABEL_TYPE == 'verb':
            label = info_dict['verb_label']
        if self.cfg.DATASET.LABEL_TYPE == 'all':
            label = {'noun_label': info_dict['noun_label'],
                     'verb_label': info_dict['verb_label'],
                     'action_label': info_dict['action_label']}
            sample_dict.update(label)
        else:
            sample_dict.update({'label': label})


        if self.use_dino_features:
            if self.features_dino is not None:
                dino_features = self.features_dino[narration]
            else:
                dino_features = np.load(self.full_feature_path_dino % narration)

            # if len(features) != len(dino_features):
            if self.use_extracted_DINO_features_only or len(dino_features) < len(features):
                clip_start, clip_end = self.start_end_clip(len(dino_features))
                uniform_frames_dino, step = np.linspace(clip_start, clip_end, frames_per_seg, endpoint=False, retstep=True)
            else:
                uniform_frames_dino = uniform_frames
            if self._split == 'train':
                step = int(step)
                # shift = np.random.choice(np.arange(int(step))) if int(step) else 0
                if int(step):
                    uniform_frames_dino = [int(i) + np.random.choice(np.arange(int(step))) for i in uniform_frames_dino]
                else:
                    uniform_frames_dino = [int(i) for i in uniform_frames_dino]
            else:
                # shift = 0
                uniform_frames_dino = [int(i + step / 2) for i in uniform_frames_dino]
            # uniform_frames_dino = [int(i) + shift for i in uniform_frames_dino]
            sample_dict['dino'] = torch.tensor(dino_features[uniform_frames_dino])

        if self.use_dino_features2:
            if self.features_dino2 is not None:
                dino_features2 = self.features_dino2[narration]
            else:
                dino_features2 = np.load(self.full_feature_path_dino2 % narration)
            if len(dino_features2) < len(dino_features):
                clip_start, clip_end = self.start_end_clip(len(dino_features2))
                uniform_frames, step = np.linspace(clip_start, clip_end, frames_per_seg, endpoint=False, retstep=True)
                if self._split == 'train':
                    if int(step):
                        uniform_frames_dino = [int(i) + np.random.choice(np.arange(int(step))) for i in uniform_frames]
                    else:
                        uniform_frames_dino = [int(i) for i in uniform_frames]
                else:
                    uniform_frames_dino = [int(i + step / 2) for i in uniform_frames]
                # uniform_frames_dino = [int(i) + np.random.choice(range(step)) for i in uniform_frames]

            dino_subset = torch.tensor(dino_features[uniform_frames_dino]).unsqueeze(0)
            dino2_subset = torch.tensor(dino_features2[uniform_frames_dino]).unsqueeze(0)
            # dino_subset = sample_dict['dino'].unsqueeze(0)
            dino_subset = torch.cat([dino_subset, dino2_subset])
            dino_dim = dino_subset.shape[-1]
            dino_subset = dino_subset.transpose(0,1).reshape(-1, dino_dim)
            sample_dict['dino'] = dino_subset
            # breakpoint()

        return sample_dict


    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        # print('Worker info', worker_info)
        # self._video_random_generator.manual_seed(32)
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self

def parse_class(x):
    if '(' not in x:
        x = " ".join(x.split('_'))
        return [x]
    x = x.split('(')
    word = x[0]
    word = " ".join(word.split('_')).strip()
    rest = x[1][:-1]
    rest = rest.split(',')
    rest = [" ".join(i.split('_')).strip() for i in rest]
    if 'plug-in' in rest:
        rest.append('plug in')
    return [word] + rest

def clip_recognition_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    classnames_and_mapping: bool = False,
    use_features: bool = False,
    split: str='train',
    cfg=None
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

        actions_idx2verbnoun = {}
        actions_verbnoun2idx = {}
        actions_idx2verbnounidx = {}
        actions_verbnounidx2idx = {}
        with open(f'{os.path.dirname(data_path)}/fho_actions_actionidx_verb_noun.txt', 'r') as actions_f:
            for line in actions_f:
                line = line.strip().split()
                action_idx = int(line[0])
                verb = line[1]
                noun = line[2]
                actions_idx2verbnoun[action_idx] = (verb, noun)
                actions_verbnoun2idx[(verb, noun)] = action_idx
                # breakpoint()
                actions_idx2verbnounidx[action_idx] = (mapping_verb_c2l[verb], mapping_noun_c2l[noun])
                actions_verbnounidx2idx[(mapping_verb_c2l[verb], mapping_noun_c2l[noun])] = action_idx

        # if classnames_and_mapping:
        if True:
            # mapping_nouns = {label: classname for label, classname in container_nouns}
            # mapping_verbs = {label: classname for label, classname in container_verbs}

            ## FIX
            def _fix_classes(mapping_nouns_label2classname):
                for k, v in mapping_nouns_label2classname.items():
                    if v.startswith('nut_'):
                        if 'food' in v:
                            mapping_nouns_label2classname[k] = 'nut'
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
                    if v.startswith('pot'):
                        if 'planter' in v:
                            mapping_nouns_label2classname[k] = 'planter'

                    v = mapping_nouns_label2classname[k]
                    mapping_nouns_label2classname[k] = v.split('(')[0]

                return mapping_nouns_label2classname

            mapping_noun_l2c = _fix_classes(mapping_noun_l2c)

            labels_nouns = list(mapping_noun_l2c.keys())
            print('LEN LABELS NOUNS', len(labels_nouns), flush=True)
            labels_nouns.sort()
            # mapping_noun_l2c = {k:v.split('(')[0] for k,v in mapping_noun_l2c.items()}

            classnames_nouns = [mapping_noun_l2c[label] for label in labels_nouns]
            print('CLASSNAMES NOUNS', classnames_nouns[:50], flush=True)
            print('CLASSNAMES NOUNS', len(classnames_nouns), flush=True)

            labels_verbs = list(mapping_verb_l2c.keys())
            print('LEN LABELS VERBS', len(labels_verbs), flush=True)
            labels_verbs.sort()
            # mapping_verb_l2c = {k: v.split('(')[0] for k, v in mapping_verb_l2c.items()}
            mapping_verb_l2c = _fix_classes(mapping_verb_l2c)
            classnames_verbs = [mapping_verb_l2c[label] for label in labels_verbs]
            print('CLASSNAMES VERBS', classnames_verbs[:50], flush=True)
            print('CLASSNAMES VERBS', len(classnames_verbs), flush=True)

            mapping_actions_l2c = {}
            for k, v in actions_verbnounidx2idx.items():
                mapping_actions_l2c[v] = f'{mapping_verb_l2c[k[0]]} {mapping_noun_l2c[k[1]]}'

            labels_actions = list(actions_verbnounidx2idx.values())
            labels_actions.sort()
            classnames_actions = [mapping_actions_l2c[label] for label in labels_actions]

            print('CLASSNAMES ACTIONS', classnames_actions[:50], flush=True)
            print('CLASSNAMES ACTIONS', len(classnames_actions), flush=True)

            addiional_output = {'mapping_nouns': mapping_noun_l2c, 'classnames_nouns': classnames_nouns,
                                'train_class_counts_noun': train_class_counts_nouns,
                                'mapping_verbs': mapping_verb_l2c, 'classnames_verbs': classnames_verbs,
                                'train_class_counts_verb': train_class_counts_verbs,
                                'mapping_actions': mapping_actions_l2c, 'classnames_actions': classnames_actions}

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

            action_label = actions_verbnoun2idx[(verb, noun)]
            narration_text = entry["narration_text"]
            if narration_text == "":
                narration_text = f"{mapping_actions_l2c[action_label]}"


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
                        "clip_uid": entry["clip_uid"],
                        "action_label": action_label,
                        "narration": narration_text
                    },
                )
            )

    else:
        raise FileNotFoundError(f"{data_path} not found.")
    if use_features:
        dataset = LabeledFeaturesDataset(
            untrimmed_clip_annotations,
            video_sampler=video_sampler,
            feature_root=video_path_prefix,
            split=split,
            cfg=cfg,
        )
    else:
        dataset = LabeledVideoDataset(
            untrimmed_clip_annotations,
            UntrimmedClipSampler(clip_sampler),
            video_sampler,
            transform,
            decode_audio=decode_audio,
            decoder=decoder,
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