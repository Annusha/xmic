import os
import os.path as osp
import pickle
import pandas as pd
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import decord
import torch

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


def fix_class_name(class_name, label_type='noun'):
    class_name = ' '.join(class_name.split('/'))
    return class_name

@DATASET_REGISTRY.register()
class EGTEA(DatasetBase):

    dataset_dir = "egtea"

    def __init__(self, cfg):
        self.cfg = cfg

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.EGTEA.ROOT))
        self.root = root
        # self.is_trimmed = cfg.DATASET.EGTEA.ROOT.IS_TRIMMED
        anno_dir = cfg.DATASET.EGTEA.ANNOTATIONS_DIR
        self.anno_dir = anno_dir
        # metadata = cfg.DATASET.EGTEA.METADATA_FILE

        self.num_clips = cfg.DATASET.EGTEA.NUM_CLIPS
        self.clip_length = cfg.DATASET.EGTEA.CLIP_LENGTH
        self.clip_stride = cfg.DATASET.EGTEA.CLIP_STRIDE
        self.sparse_sample = cfg.DATASET.EGTEA.SPARSE_SAMPLE

        # egtea_video_list = torch.load(os.path.join(anno_dir, 'egtea_video_list.pth.tar'))
        with open(os.path.join(anno_dir, 'video_lens.json'), 'r') as f:
            len_dict = json.load(f)
        # breakpoint()



        self.train_class_counts, self.dist_splits = self.create_lt_splits(cfg, None)
        lab2cname, classes = self.get_lab2cname()

        val_samples = []
        label_type = cfg.DATASET.LABEL_SUBTYPES.split(',')[0]

        with open(self.anno_dir + "val_allTrainWoAllTest.txt", 'r') as f:
            for line in f:
                line = line.strip().split()
                if label_type == 'noun' and len(line) <= 3:
                    continue
                clip_id = line[0]
                video_id = '-'.join(clip_id.split('-')[:3])
                vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))

                if label_type == 'noun':
                    label = int(line[3])
                elif label_type == 'verb':
                    label = int(line[2])


                sample_dict = {
                    'full_path': vid_fullpath,
                    'label': label,
                    'start_frame': 0,
                    'end_frame': len_dict[clip_id],
                    'rel_path': vid_relpath,
                    'clip_id': clip_id
                }

                val_samples.append(sample_dict)

        test_samples = []
        label_type = cfg.DATASET.LABEL_SUBTYPES.split(',')[0]
        with open(self.anno_dir + "test_split_all.txt", 'r') as f:
            for line in f:
                line = line.strip().split()
                if label_type == 'noun' and len(line) <= 3:
                    continue
                clip_id = line[0]
                video_id = '-'.join(clip_id.split('-')[:3])
                vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))

                if label_type == 'noun':
                    label = int(line[3])
                elif label_type == 'verb':
                    label = int(line[2])
                sample_dict = {
                    'full_path': vid_fullpath,
                    'label': label,
                    'start_frame': 0,
                    'end_frame': len_dict[clip_id],
                    'rel_path': vid_relpath,
                    'clip_id': clip_id
                }

                test_samples.append(sample_dict)


        self.n_test_classes = len(classes)
        self.test_classes = classes
        _, self.dist_splits_novel_base = self.create_base_novel_splits(cfg, test_samples, lab2cname=lab2cname)

        super().__init__(train_x=[], val=val_samples, test=test_samples)



    def get_num_classes(self, data_source=None):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        # noun, verb, action
        # return [300, 97, 3806]
        # return {'noun': 53, 'verb': 19, 'action': 106}
        return {'noun': 54, 'verb': 20, 'action': 106}


    def get_lab2cname(self, data_source=None, seen_subset=True):
        lab2cname = {}
        classnames = {}
        for label_type in ['verb', 'noun']:
            if label_type not in self.cfg.DATASET.LABEL_SUBTYPES: continue
            mappping_split = 'noun_idx.txt' if label_type == 'noun' else 'verb_idx.txt'

            mapping_path = os.path.join(self.anno_dir, mappping_split)

            lab2cname[label_type] = {0: ''}
            classnames[label_type] = ['']

            with open(mapping_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    line = line.split()
                    cur_idx = line[-1]
                    cname = ' '.join(line[:-1])
                    cur_idx = int(cur_idx)
                    cname = fix_class_name(class_name=cname, label_type=label_type)
                    classnames[label_type].append(cname)
                    lab2cname[label_type][cur_idx] = cname

        # classnames = list(classnames)
        for k in classnames.keys():
            classnames[k] = list(classnames[k])
            print(f'CLASSNAMES {k}', classnames[k][:50], flush=True)
            print(f'CLASSNAMES {k}', len(classnames[k]), flush=True)
            print(f'LEN LABELS {k}', len(lab2cname[k]), flush=True)
        return lab2cname, classnames

    def create_lt_splits(self, cfg, data_source):
        return None, None



    def create_base_novel_splits(self, cfg, data_source, lab2cname):
        return None, None




def get_raw_item(sample, is_training=True, num_clips=1, clip_length=32, clip_stride=2, sparse_sample=False,
                 narration_selection='random', root=''):

    vid_path  = sample['full_path']
    start_frame = sample['start_frame']
    end_frame = sample['end_frame']
    # sentence = sample['narration']

    if is_training:
        assert num_clips == 1
        if end_frame < clip_length * clip_stride:
            frames = video_loader_by_frames(vid_path, list(np.arange(0, end_frame)))
            zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
            frames = torch.cat((frames, zeros), dim=0)
            frames = frames[::clip_stride]
        else:
            start_id = np.random.randint(0, end_frame - clip_length * clip_stride + 1)
            frame_ids = np.arange(start_id, start_id + clip_length * clip_stride, clip_stride)
            frames = video_loader_by_frames(vid_path, frame_ids)
    else:
        if clip_length == -1:
            frames = video_loader_by_frames(vid_path, list(np.arange(0, end_frame)))
        else:
            if end_frame < clip_length * clip_stride:
                frames = video_loader_by_frames(vid_path, list(np.arange(0, end_frame)))
                zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                frames = torch.cat((frames, zeros), dim=0)
                frames = frames[::clip_stride]
                frames = frames.repeat(num_clips, 1, 1, 1)
            else:
                frame_ids = []
                for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                    frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                frames = video_loader_by_frames(vid_path, frame_ids)
    return frames, ''


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
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)),
                                      num_segments=clip_length, jitter=jitter)
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


def video_loader_by_frames(vid, frame_ids):
    vr = decord.VideoReader(vid)
    try:
        frames = vr.get_batch(frame_ids)
        # breakpoint()
        if torch.is_tensor(frames):
            frames = torch.tensor(frames).to(torch.float32)
        else:
            frames = torch.tensor(frames.asnumpy()).to(torch.float32)
        frames = [frame for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        # frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
        raise EOFError('Bad video')
    return torch.stack(frames, dim=0)



def generate_label_map(action_idx_file):
    labels = []
    with open(action_idx_file) as f:
        for row in f:
            row = row.strip()
            narration = ' '.join(row.split(' ')[:-1])
            labels.append(narration.replace('_', ' ').lower())
            # labels.append(narration)
    mapping_vn2act = {label: i for i, label in enumerate(labels)}

    return labels, mapping_vn2act

