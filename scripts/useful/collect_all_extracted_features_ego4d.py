import pickle
import pandas as pd
import os
import tqdm
import numpy as np
import json


exp = '39_ego4d_extract_clip_vitl14_HC_fixed_len'
dataset = 'Ego4DRecognitionWrapper' # EpicKitchenSegmentsSpecialOCv2  # EpicKitchenSegments
root = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/'
# mappping_split = 'EPIC_100_noun_classes.csv'
# split_file = 'val'
split_file = 'train'
epic_path = '/mnt/graphics_ssd/nimble/users/annakukleva/data/ego4d/annotations/fho_lta_%s.json' % split_file

output_file = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/{split_file}.pkl'

problems_file = f'/mnt/graphics_ssd/nimble/users/annakukleva/output/multimodal-prompt-learning/clip_feat/{dataset}/{exp}/problems_{split_file}.txt'


output = {}
problems = []

with open(epic_path, 'r') as f:
    annotations = json.load(f)
annotations = annotations['clips']

for row_segment in tqdm.tqdm(
    annotations,
    f"Populating Dataset",
    total=len(annotations),
):
    narration_id = f'{split_file}_{row_segment["clip_uid"]}_{row_segment["action_idx"]}'
    local_path = root + f'segments/{narration_id}.npy'
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
