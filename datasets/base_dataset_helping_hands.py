import os
import pdb

import tqdm
import random
from abc import abstractmethod

import av
import cv2
import decord
import ffmpeg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from iopath.common.file_io import g_pathmgr


class TextVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 meta_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 neg_param=None,
                 crop_w_boxes=False,
                 video_res=256,
                 slice_idx=0,
                 n_slice=1,
                 hard_neg=False,
                 extract_features=False,
                 ):
        self.hard_neg = hard_neg
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        self.crop_with_boxes = crop_w_boxes
        self.extract_features = extract_features

        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if meta_dir is not None:
            self.meta_dir = os.path.expandvars(meta_dir)
        else:
            self.meta_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self.neg_param = neg_param
        self.crop_w_boxes = crop_w_boxes
        self.video_res = video_res
        self.slice_idx = slice_idx
        self.n_slice = n_slice
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        # return 1000
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')
        # frame_sample = 'rand'
        frame_sample = self.video_params.get('frame_sample', 'rand')

        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                if video_loading == 'verbose':
                    print(f'Video loading error {e}, replace with zeros')
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

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

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data



def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def sample_frames_clips(start, end, vlen, acc_samples):
    start = max(0, start)
    end = min(vlen, end)

    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges
                      ]
    return frame_idxs


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


def sample_frames_start_end(num_frames, start, end, sample='rand', fix_start=None):
    acc_samples = min(num_frames, end)
    if end - start + 1 == num_frames:
        intervals = np.linspace(start=start, stop=end + 1, num=acc_samples + 1).astype(int)
    else:
        intervals = np.linspace(start=start, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        # frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        frame_idxs = []
        for x in ranges:
            if x[1] == x[0]:
                frame_idxs.append(x[0])
            else:
                frame_idxs.append(random.choice(range(x[0], x[1])))
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs



def read_frames_cv2_egoclip_decord(vpath, start_second, end_second=None, chunk_len=600, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        vr = decord.VideoReader(vpath)
        second_offset = start_second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(start_second) // chunk_len * chunk_len
        second_offset = start_second - chunk_start
        vr = decord.VideoReader(vpath)
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - start_second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= start_second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)),
                                      num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    # breakpoint()
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids)
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids))
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1)
            chankid = int(vpath.split('/')[-1].split('.mp4')[0])
            vpath2 = f'{os.path.dirname(vpath)}/{chankid+1}.mp4'
            vr2 = decord.VideoReader(vpath2)
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2)
            # breakpoint()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
            frames = torch.tensor(frames)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids)
    frames = frames.to(torch.float32) / 255
    return frames.permute(0, 3, 1, 2), [f / 30 for f in frame_ids]



def read_frames_cv2_epic(video_path, start_frame, stop_frame, num_frames, sample='rand', fix_start=None,
                         high_res=False):
    # get indexes of sampled frames
    frame_idxs = sample_frames_start_end(num_frames, start_frame, stop_frame, sample=sample, fix_start=fix_start)
    # frame_idxs = get_frame_ids(start_frame,stop_frame,num_frames)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        if high_res:
            img_name = str(index) + '.jpg'
        else:
            img_name = 'frame_' + str(index).zfill(10) + '.jpg'
        frame = cv2.imread(os.path.join(video_path, img_name))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame)
        # (H x W x C) to (C x H x W)
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
        success_idxs.append(index)

    frames = torch.stack(frames).float() / 255
    return frames, success_idxs



def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def read_frames_sth(frame_paths, num_frames, sample='rand', fix_start=None, retry=3):
    vlen = len(frame_paths)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    try:
        frames = []
        for idx in frame_idxs:
            with g_pathmgr.open(frame_paths[idx], "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(img))
    except:
        raise Exception("Failed to load images {}".format(frame_paths[0]))
    frames = torch.stack(frames).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


decord.bridge.set_bridge("torch")


def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def read_frames_decord_start_end(video_path, start, end, num_frames):
    video_reader = decord.VideoReader(video_path, num_threads=1)

    vlen = len(video_reader)
    frame_idxs = sample_frames_clips(start, end, vlen, num_frames + 1)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    'av': read_frames_av,
    'cv2_epic': read_frames_cv2_epic,
    'cv2_egoclip': read_frames_cv2_egoclip_decord,
    'cv2_sth': read_frames_sth,
    'decord': read_frames_decord,
    'decord_start_end': read_frames_decord_start_end,
}