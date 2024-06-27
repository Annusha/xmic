import itertools
import os

import torch
import torch.utils.data
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
)

# from .build import DATASET_REGISTRY
from datasets.ego4d_helper import *
# from . import ptv_dataset_helper
# from ..utils import logging, video_transformer

# logger = logging.get_logger(__name__)


from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from typing import Dict, Any

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


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
class Ego4dRecognition(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode=='test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')

        self.dataset = ptv_dataset_helper.clip_recognition_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
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
        return value

    def __len__(self):
        return self.dataset.num_videos
