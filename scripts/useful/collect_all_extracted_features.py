import pickle
import pandas as pd
import os
import tqdm
import numpy as np


exp = '40_extract_lavila_videos_vitb16_segments_16frames_fixed_norm_and_crop'
dataset = 'EpicKitchenSegments' # EpicKitchenSegmentsSpecialOCv2  # EpicKitchenSegments
root = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/'
mappping_split = 'EPIC_100_noun_classes.csv'
epic_path = '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/epic_fadime/'
split_file = 'train'
# split_file = 'validation'

output_file = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/{split_file}.pkl'

problems_file = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/problems_{split_file}.txt'


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

epic_train = pd.read_csv(os.path.join(epic_path, 'annotations', f'EPIC_100_{split_file}.csv'), header=0, low_memory=False, names=name_list)

output = {}
problems = []

for _, row_segment in tqdm.tqdm(
    epic_train.iterrows(),
    f"Populating Dataset",
    total=len(epic_train),
):
    narration_id = row_segment.narration_id
    local_path = root + f'segments_npy/{split_file}_{narration_id}.npy'
    if os.path.exists(local_path):
        try:
            data = np.load(local_path)
        except Exception as e:
            print(e, local_path)
            problems.append(narration_id)
            continue
        # with open(local_path, 'rb') as f:
        #     data = pickle.load(f)

        # for key, val in data.items():
        output[narration_id] = data
    else:
        problems.append(narration_id)
        print(f'Does not exist: {local_path}')

print(f'{split_file}: {len(output)}')

print('Problems')
print(problems)

with open(problems_file, 'w') as f:
    for problem_name in problems:
        f.write(f'{problem_name}\n')

if problems == []:
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)
    print('saved', output_file)
