import os
import pickle
import pandas as pd
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

# from epic_kitchens.hoa import load_detections, CroppingRenderer

# from .oxford_pets import OxfordPets

# class Datum2(Datum):
#     def __init__(self, impath, label, label2):
#         super().__init__(impath=impath)
#         self._label = label1 = label2


def fix_class_name(class_name, label_type='noun'):
    if label_type == 'action':
        class_name = class_name.split()
        verb = fix_class_name(class_name[0])
        noun = fix_class_name(class_name[1])
        return f'{verb} {noun}'.strip()
    if label_type == 'action-ek55':
        class_name = class_name.split('_')
        verb = fix_class_name(class_name[0])
        noun = fix_class_name(class_name[1])
        return f'{verb} {noun}'.strip()
    class_name = class_name.split(':')
    class_name.reverse()
    class_name = ' '.join(class_name)
    class_name = class_name.split('-')
    class_name = ' '.join(class_name)
    if class_name == 'use to':
        class_name = 'use'
    if class_name == 'end of':
        class_name = 'end'
    if class_name == 'bring into':
        class_name = 'bring'
    return class_name

@DATASET_REGISTRY.register()
class EpicKitchenSegmentsAllLabelTypes(DatasetBase):

    dataset_dir = "epic_kitchen_segments"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.cfg = cfg
        self.dataset_dir = root
        self.dataset_detection_root = cfg.DATASET.DETECTION_ROOT

        self.image_dir = root
        self.label_type = cfg.DATASET.LABEL_TYPE
        self.subset = cfg.DATASET.SUBSET
        # self.preprocessed = os.path.join(self.dataset_dir, f"preprocessed_{self.subset}_{self.label_type}.pkl")
        assert self.label_type in ['all', 'all_w_narrations'] # ['noun', 'verb', 'action']
        self.full_path = os.path.join(self.image_dir, 'epic-kitchens', '%s', 'rgb_frames', '%s')  # participant_id, video_id
        self.full_path_video = os.path.join(self.image_dir, 'epic_kitchens_videos_256ss', '%s', 'videos', '%s.MP4')  # resized videos
        self.full_feature_path = os.path.join(self.image_dir, 'features', cfg.DATALOADER.FEATURES_MODEL, 'EpicKitchenSegments', cfg.DATALOADER.FEATURES_NAME, 'segments_npy', '%s_%s.npy')  # split narration_id
        self.full_detection_path = os.path.join(self.dataset_detection_root, '3l8eci2oqgst92n14w2yqi5ytu/hand-objects', '%s', '%s.pkl')  # participant_id, video_id
        self.save_hands_crops_file = cfg.DATALOADER.CROPPER.SAVE_FILE

        self.hand_crops = cfg.DATALOADER.HAND_CROPS
        self.frame_type = cfg.DATALOADER.FRAME_TYPE
        if self.hand_crops:
            self.crop_file_name = f'hand_thd{cfg.DATALOADER.CROPPER.HAND_THS}_obj_thd{cfg.DATALOADER.CROPPER.OBJ_THS}{"_ONLY_inter_obj" if cfg.DATALOADER.CROPPER.ONLY_INTERACTED_OBJ else ""}{"_with_HANDS" if cfg.DATALOADER.CROPPER.WITH_HANDS else ""}{cfg.DATALOADER.CROPPER.TAG}'

        if cfg.TEST.CROSS_DATASET.EVAL and "Epic" in cfg.TEST.CROSS_DATASET.DATASET_NAME:
            val = self.read_data("EPIC_100_validation.csv")
            train = val
            test = val
        else:
            train = self.read_data("EPIC_100_train.csv")
            val = self.read_data("EPIC_100_validation.csv")
            test = val
        self.train_classes, self.test_classes = None, None
        self.n_test_classes = 0


        self.train_class_counts, self.dist_splits = self.create_lt_splits(cfg, train)
        lab2cname, classes = self.get_lab2cname()
        if self.n_test_classes == 0:
            self.n_test_classes = {}
            for k,v in classes.items():
                self.n_test_classes[k] = len(v)
            self.test_classes= classes

        _, self.dist_splits_novel_base = self.create_base_novel_splits(cfg, train, lab2cname=lab2cname)


        super().__init__(train_x=train, val=val, test=test)

        self.text_val = self.read_text_source()



    def read_text_source(self, split='val'):
        root  = os.path.join(self.image_dir, 'annotations')
        texts = pd.read_csv(root + '/retrieval_annotations/EPIC_100_retrieval_test_sentence.csv').values[:, 1]
        return texts


    def read_data(self, split_file, cname2lab=None, seen_subset=True):

        name_list = [
            "narration_id",
            "participant_id",
            "video_id",
            "narration_timestamp",
            "start_timestamp",
            "stop_timestamp",
            "start_frame",
            "stop_frame",
            "narration",
            "verb",
            "verb_class",
            "noun",
            "noun_class",
            "all_nouns",
            "all_noun_classes",
        ]

        epic_train = pd.read_csv(os.path.join(self.image_dir, 'annotations', split_file), header=0, low_memory=False, names=name_list)

        name_list_info = ['video_id', 'duration', 'fps', 'resolution']
        epic_video_info = pd.read_csv(os.path.join(self.image_dir, 'annotations', 'EPIC_100_video_info.csv'), header=0, low_memory=False,
                                 names=name_list_info)

        with open(os.path.join(self.image_dir, 'annotations', 'EPIC_100_rgb_frames_info.json'), 'r') as f:
            epic_frames_info = json.load(f)

        fps_dict = {}
        duration_dict = {}
        for i in range(len(epic_video_info['video_id'])):
            video_id_cur = epic_video_info['video_id'].iloc[i]
            fps_dict[video_id_cur] = epic_video_info.loc[epic_video_info['video_id'] == video_id_cur, 'fps'].iloc[0]
            duration_dict[video_id_cur] = epic_video_info.loc[epic_video_info['video_id'] == video_id_cur, 'duration'].iloc[0]

        epic_mapping = {}
        for label_type in ['verb', 'noun']:
            mappping_split = 'EPIC_100_noun_classes.csv' if label_type == 'noun' else 'EPIC_100_verb_classes.csv'


            epic_mapping_csv = pd.read_csv(os.path.join(self.image_dir, 'annotations', mappping_split))
            # as the instences items looks as "['tap', 'tap:water', 'water:tap', 'flow:tap', 'tab', 'tap:hot']"
            # we need to convert it to ['tap', 'tap:water', 'water:tap', 'flow:tap', 'tab', 'tap:hot']
            epic_mapping_csv['instances'] = epic_mapping_csv['instances'].apply(eval)
            epic_mapping[label_type] = {}
            # subclassnames = []
            # subclass_iname2idx = {}
            for item in epic_mapping_csv.iterrows():
                for subitem in item[1].instances:
                    subitem_fixed_name = fix_class_name(subitem)
                    epic_mapping[label_type][subitem_fixed_name] = fix_class_name(item[1].key)


        label_type = 'action'
        # subclass_iname2idx = None
        # self._subclassnames = []
        mappping_split = 'actions.csv'
        epic_mapping_csv = pd.read_csv(os.path.join(self.image_dir, 'annotations', mappping_split),
                                       names=["id", "verb", "noun", "action"], header=0, low_memory=False)
        epic_mapping[label_type] = {}
        action_label_mapping = {}
        for item in epic_mapping_csv.iterrows():
            action_label_mapping[(item[1].verb, item[1].noun)] = item[1].id
            epic_mapping[label_type][item[1].id] = fix_class_name(item[1].action, label_type='action')


        # place segments in a list
        # all_segment_dict = defaultdict(list)
        all_segments_list = []
        max_frame = 0
        all_crops = {}

        if self.hand_crops:
            crop_path = os.path.join(self.image_dir, 'annotations', 'hand_crops', f'{self.crop_file_name}_{split_file}.pkl')

            if os.path.exists(crop_path):
                with open(crop_path, 'rb') as f:
                    all_crops = pickle.load(f)
            else:
                raise FileNotFoundError(f'Hand crops {crop_path}')


        total_n_frames = 0
        frames_per_segment = []
        total_secs = 0
        secs_per_segment = []

        for _, row_segment in tqdm(
            epic_train.iterrows(),
            f"Populating Dataset {split_file}",
            total=len(epic_train),
        ):

            segment = dict()
            segment["video_id"] = row_segment.video_id
            segment["start_frame"] = row_segment.start_frame
            segment["end_frame"] = row_segment.stop_frame
            segment["verb"] = fix_class_name(row_segment.verb)
            segment["verb_class"] = row_segment.verb_class
            segment["noun"] = fix_class_name(row_segment.noun)
            segment["noun_class"] = row_segment.noun_class
            segment["all_nouns"] = row_segment.all_nouns
            segment["narration"] = row_segment.narration
            segment["narration_id"] = row_segment.narration_id
            fps = epic_frames_info[row_segment.video_id] / duration_dict[row_segment.video_id]
            segment["fps"] = fps # fps_dict[row_segment.video_id]
            segment["duration"] = duration_dict[row_segment.video_id]
            segment["total_frame_number"] = epic_frames_info[row_segment.video_id]
            segment['start_timestamp'] = row_segment.start_timestamp
            segment['stop_timestamp'] = row_segment.stop_timestamp
            segment['path'] = self.full_path % (row_segment.participant_id, row_segment.video_id) + '/frame_%010d.jpg'
            segment['video_path'] = self.full_path_video % (row_segment.participant_id, row_segment.video_id)
            segment['participant_id'] = row_segment.participant_id
            segment['feature_path'] = self.full_feature_path % (split_file, segment["narration_id"])
            if row_segment.video_id in ['P0109', 'P27_103']:
                segment['dection_path'] = f'{self.dataset_detection_root}/{row_segment.video_id}.pkl'
            else:
                segment['dection_path'] = self.full_detection_path % (row_segment.participant_id, row_segment.video_id)

            frames_per_segment.append(row_segment.stop_frame - row_segment.start_frame + 1)
            total_n_frames += (row_segment.stop_frame - row_segment.start_frame + 1)

            def _string2sec(timestamp):
                timestamp = timestamp.split(':')
                hours = int(timestamp[0])*60*60 # hours convert to secs
                mins = int(timestamp[1]) * 60 # mins to secs
                secs = float(timestamp[2])
                return hours + mins + secs

            stop_timestamp  = _string2sec(row_segment.stop_timestamp)
            start_timestamp = _string2sec(row_segment.start_timestamp)
            secs_per_segment.append(stop_timestamp - start_timestamp)
            total_secs += (stop_timestamp - start_timestamp)

            for label_type in ['noun', 'verb']:
                classname = epic_mapping[label_type][segment[label_type]]
                segment[f'mapped_{label_type}'] = classname

            label_type = 'action'
            segment["action_class"] = action_label_mapping[(segment["verb_class"], segment["noun_class"])]
            classname = epic_mapping[label_type][segment["action_class"]]
            segment[f'mapped_{label_type}'] = classname



            if row_segment.stop_frame > max_frame:
                max_frame = row_segment.stop_frame


            if self.hand_crops:
                if segment['narration_id'] in all_crops:
                    segment['crop_bb'] = all_crops[segment['narration_id']]

            all_segments_list.append(segment)


        print("Total Number of Frames: ", total_n_frames)
        print("Min, Max and Average Number of frames: ", np.min(frames_per_segment), np.max(frames_per_segment), np.mean(frames_per_segment))

        print("Total Number of Secs: ", total_secs )
        print("Min, Max and Average Number of secs: ", np.min(secs_per_segment), np.max(secs_per_segment), np.mean(secs_per_segment))

        return all_segments_list



    def get_num_classes(self, data_source=None):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        # noun, verb, action
        # return [300, 97, 3806]
        return {'noun': 300, 'verb': 97, 'action': 3806}

    def get_lab2cname(self, data_source=None, seen_subset=True):
        lab2cname = {}
        classnames = {}
        for label_type in ['verb', 'noun']:
            if label_type not in self.cfg.DATASET.LABEL_SUBTYPES: continue
            mappping_split = 'EPIC_100_noun_classes.csv' if label_type == 'noun' else 'EPIC_100_verb_classes.csv'

            epic_mapping_csv = pd.read_csv(os.path.join(self.image_dir, 'annotations', mappping_split))
            # epic_mapping_csv = pd.read_csv(os.path.join(image_dir, 'annotations', mappping_split))
            # as the instences items looks as "['tap', 'tap:water', 'water:tap', 'flow:tap', 'tab', 'tap:hot']"
            # we need to convert it to ['tap', 'tap:water', 'water:tap', 'flow:tap', 'tab', 'tap:hot']
            epic_mapping_csv['instances'] = epic_mapping_csv['instances'].apply(eval)
            lab2cname[label_type] = {}
            classnames[label_type] = []


            for item in epic_mapping_csv.iterrows():
                tmp_name = fix_class_name(item[1].key)
                # if seen_nouns is None or tmp_name in seen_nouns:
                cur_idx = item[1].id
                lab2cname[label_type][cur_idx] = fix_class_name(item[1].key)
                classnames[label_type].append(tmp_name)


        label_type = 'action'
        if label_type in self.cfg.DATASET.LABEL_SUBTYPES:
            mappping_split = 'actions.csv'
            lab2cname[label_type] = {}
            classnames[label_type] = []
            epic_mapping_csv = pd.read_csv(os.path.join(self.image_dir, 'annotations', mappping_split), names=["id", "verb", "noun", "action"], header=0, low_memory=False)
            for item in epic_mapping_csv.iterrows():
                tmp_name = fix_class_name(item[1].action, label_type='action')
                lab2cname[label_type][item[1].id] = tmp_name
                classnames[label_type].append(tmp_name)

        # classnames = list(classnames)
        for k in classnames.keys():
            classnames[k] = list(classnames[k])
            print(f'CLASSNAMES {k}', classnames[k][:50], flush=True)
            print(f'CLASSNAMES {k}', len(classnames[k]), flush=True)
            print(f'LEN LABELS {k}', len(lab2cname[k]), flush=True)
        return lab2cname, classnames

    def create_lt_splits(self, cfg, data_source):
        if not cfg.TEST.LT_EVAL and not cfg.TRAINER.BALANCED_CE:
            return None, None

        num_classes = self.get_num_classes()
        # num_classes = {'noun': num_classes_list[0], 'verb': num_classes_list[1], 'action': num_classes_list[2]}
        head_split = {} # []
        tail_split = {} #[]
        fewshot_split = {} #[]
        train_class_counts = {}
        for label_type in ['noun', 'verb', 'action']:
            train_class_counts[label_type] = defaultdict(int)
            for item in data_source:
                train_class_counts[label_type][item[f'{label_type}_class']] += 1


            if len(train_class_counts[label_type]) < num_classes[label_type]:
                for class_idx in range(num_classes[label_type]):
                    if class_idx in train_class_counts[label_type]:
                        continue
                    else:
                        train_class_counts[label_type][class_idx] = 0

            head_split[label_type] =  []
            tail_split[label_type] = []
            fewshot_split[label_type] = []
            cumulative_sum = 0
            half_total_sum = len(data_source) / 2
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


    def create_base_novel_splits(self, cfg, data_source, lab2cname):
        # if not cfg.TEST.BASE_NOVEL_EVAL:
        #     return None, None

        shared_split_exact = {}
        unique_split_exact = {}
        shared_split_semantic = {}
        unique_split_semantic = {}
        shared_split_semantic_wo_exact = {}
        train_class_counts = {}
        num_classes = self.get_num_classes()
        for label_type in ['noun', 'verb']:

            if label_type not in lab2cname: continue
            classname2label = {v:k for k,v in lab2cname[label_type].items()}
            print(len(classname2label), len(lab2cname[label_type]))
            assert len(classname2label) == len(lab2cname[label_type])


            shared_classnames_exact = [] # epic_ego_common_exact_noun.txt
            with open(os.path.join(self.cfg.DATASET.ROOT, 'annotations', f'epic_ego_common_exact_{label_type}.txt'), 'r') as f:
                for line in f:
                    shared_classnames_exact.append(" ".join(line.strip().split('_')))

            unique_classnames_exact = []
            with open(os.path.join(self.cfg.DATASET.ROOT, 'annotations', f'epic_ego_non_common_exact_{label_type}.txt'), 'r') as f:
                for line in f:
                    unique_classnames_exact.append(" ".join(line.strip().split('_')))

            shared_split_exact[label_type] = [classname2label[base_name] for base_name in shared_classnames_exact]
            unique_split_exact[label_type] = [classname2label[novel_name] for novel_name in unique_classnames_exact]

            print(f'SHARED_EXACT_SPLIT {label_type}', len(shared_split_exact[label_type]), shared_split_exact[label_type], flush=True)
            print(f'UNIQUE_EXACT_SPLIT {label_type}', len(unique_split_exact[label_type]), unique_split_exact[label_type], flush=True)


            shared_classnames_semantic = [] # epic_ego_common_exact_noun.txt
            with open(os.path.join(self.cfg.DATASET.ROOT, 'annotations', f'epic_ego_common_semantic_{label_type}.txt'), 'r') as f:
                for line in f:
                    shared_classnames_semantic.append(" ".join(line.strip().split('_')))

            shared_classnames_semantic_wo_exact = []  # epic_ego_common_exact_noun.txt
            for class_name in shared_classnames_semantic:
                if class_name in shared_classnames_exact:
                    continue
                else:
                    shared_classnames_semantic_wo_exact.append(class_name)

            unique_classnames_semantic = []
            with open(os.path.join(self.cfg.DATASET.ROOT, 'annotations', f'epic_ego_non_common_semantic_{label_type}.txt'), 'r') as f:
                for line in f:
                    unique_classnames_semantic.append(" ".join(line.strip().split('_')))

            shared_split_semantic[label_type] = [classname2label[base_name] for base_name in shared_classnames_semantic]
            unique_split_semantic[label_type] = [classname2label[novel_name] for novel_name in unique_classnames_semantic]
            shared_split_semantic_wo_exact[label_type] = [classname2label[novel_name] for novel_name in shared_classnames_semantic_wo_exact]

            print(f'SHARED_SEMANTIC_SPLIT {label_type}', len(shared_split_semantic[label_type]), shared_split_semantic[label_type], flush=True)
            print(f'UNIQUE_SEMANTIC_SPLIT {label_type}', len(unique_split_semantic[label_type]), unique_split_semantic[label_type], flush=True)
            print(f'shared_split_semantic_wo_exact {label_type}', len(shared_split_semantic_wo_exact[label_type]), shared_split_semantic_wo_exact[label_type],
                  flush=True)

            train_class_counts[label_type] = defaultdict(int)
            for item in data_source:
                train_class_counts[label_type][item[f'{label_type}_class']] += 1


            if len(train_class_counts[label_type]) < num_classes[label_type]:
                for class_idx in range(num_classes[label_type]):
                    if class_idx in train_class_counts[label_type]:
                        continue
                    else:
                        train_class_counts[label_type][class_idx] = 0

        return train_class_counts, [shared_split_exact, unique_split_exact, shared_split_semantic, unique_split_semantic, shared_split_semantic_wo_exact]

        return None, None


