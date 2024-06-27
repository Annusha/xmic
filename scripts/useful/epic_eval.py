
import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('../C1-Action-Recognition/')
from evaluate import script_main

# save_path = osp.join(path, f"{local_rank}_{cfg.DATASET.LABEL_TYPE}.pkl")
def collect_data(verb_path, noun_path, pref, output_pref):
    output_path = os.path.join(output_pref, f'{pref}_results.pkl')
    # if not os.path.exists(output_path):
    if True:
        noun_paths = []
        verb_paths = []
        for root, dirs, files in os.walk(verb_path):
            for file_name in files:
                if not file_name.endswith('pkl'): continue
                if 'verb' in file_name: verb_paths.append(os.path.join(root, file_name))

        for root, dirs, files in os.walk(noun_path):
            for file_name in files:
                if not file_name.endswith('pkl'): continue
                if 'noun' in file_name: noun_paths.append(os.path.join(root, file_name))

        all_nouns = defaultdict(dict)
        for file_name in noun_paths:
            with open(file_name, 'rb') as f:
                nouns_part = pickle.load(f)
            for item in nouns_part:
                all_nouns[item['narration_id']]['noun_output'] = item['noun_output'].cpu().numpy()

        all_verbs = defaultdict(dict)
        for file_name in verb_paths:
            with open(file_name, 'rb') as f:
                verb_part = pickle.load(f)
            for item in verb_part:
                all_verbs[item['narration_id']]['verb_output'] = item['verb_output'].cpu().numpy()

        results = []
        assert all_nouns.keys() == all_verbs.keys()
        for narration_id in all_nouns.keys():
            results.append({
                'narration_id': narration_id,
                'noun_output': all_nouns[narration_id]['noun_output'],
                'verb_output': all_verbs[narration_id]['verb_output']
            })


        print('LEN results', len(results), flush=True)
        print('output path', output_path, flush=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

    root = Path('/mnt/graphics_ssd/nimble/users/annakukleva/code/multimodal-prompt-learning/')
    script_main(**{
        'results': output_path,
        'labels': '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/annotations/EPIC_100_validation.pkl',
        'tail_verb_classes_csv': '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/annotations/EPIC_100_tail_verbs.csv',
        'tail_noun_classes_csv': '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/annotations/EPIC_100_tail_nouns.csv',
        'unseen_participant_ids_csv': '/mnt/graphics_ssd/nimble/users/annakukleva/data/epic/annotations/EPIC_100_unseen_participant_ids_validation.csv',
        'root': root
        })



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_verb", default='TemporalClipFT', type=str)
    parser.add_argument("--trainer_noun", default='TemporalClipFT', type=str)
    parser.add_argument("--cfg_verb", type=str, default="16_temp_ft_clip_baseline_vit_b16_segm")
    parser.add_argument("--cfg_noun", type=str, default="16_temp_ft_clip_baseline_vit_b16_segm")
    args = parser.parse_args()

    # collect_data(
    #     verb_path=f'output/EpicKitchenSegments/{args.trainer_verb}/{args.cfg_verb}',
    #     noun_path=f'output/EpicKitchenSegments/{args.trainer_noun}/{args.cfg_noun}',
    #     pref=f'V_{args.trainer_verb}_{args.cfg_verb}_N_{args.trainer_noun}_{args.cfg_noun}',
    #     output_pref=f'output/EpicKitchenSegments/')

    collect_data(
        verb_path=f'output/ZeroshotCLIPSegments/04_epic100_clip_zs_vit_b16/EpicKitchenSegments/_verb/',
        noun_path=f'output/ZeroshotCLIPSegments/04_epic100_clip_zs_vit_b16/EpicKitchenSegments/_noun/',
        pref=f'zeroshot',
        output_pref=f'output/ZeroshotCLIPSegments/04_epic100_clip_zs_vit_b16/EpicKitchenSegments/')


# output/ZeroshotCLIPSegments/04_epic100_clip_zs_vit_b16/EpicKitchenSegments/_verb/0_verb.pkl