import os
import json
from collections import defaultdict


label_type = 'noun'
ano_root = '/mnt/graphics_ssd/nimble/users/annakukleva/data/ego4d/annotations/'
split = 'train'
ano_file = f'fho_lta_{split}.json'  # fho_lta_val.json



output  = ano_root + f'mappping_%s.txt'

with open(os.path.join(ano_root, ano_file), 'r') as f:
    train_data = json.load(f)

split = 'val'
ano_file = f'fho_lta_{split}.json'  # fho_lta_val.json
with open(os.path.join(ano_root, ano_file), 'r') as f:
    val_data = json.load(f)


container_nouns = set()
container_verbs = set()
mapping = defaultdict(set)
mapping_label = 'verb'
for entry in train_data['clips']:
    # noun = clip['noun'].split('_')[0]
    # noun_label = clip['noun_label']
    # verb = clip['verb'].split('_')[0]
    # verb_label = clip['verb_label']

    # container_nouns.add(entry['noun'].split('(')[0])
    # container_verbs.add(entry['verb'].split('(')[0])

    container_nouns.add(entry['noun'])
    container_verbs.add(entry['verb'])

    mapping[entry[mapping_label].split('(')[0]].add(entry[mapping_label])

for entry in val_data['clips']:
    # noun = clip['noun'].split('_')[0]
    # noun_label = clip['noun_label']
    # verb = clip['verb'].split('_')[0]
    # verb_label = clip['verb_label']

    # container_nouns.add(entry['noun'].split('(')[0])
    # container_verbs.add(entry['verb'].split('(')[0])

    container_nouns.add(entry['noun'])
    container_verbs.add(entry['verb'])

    mapping[entry[mapping_label].split('(')[0]].add(entry[mapping_label])


print(len(container_nouns))
print(len(container_verbs))

for k,v in mapping.items():
    if len(v) > 1:
        print(k, v)

with open(output % 'noun', 'w') as f:
    for class_idx, class_name in enumerate(container_nouns):
        f.write(f'{class_idx} {class_name}\n')


with open(output % 'verb', 'w') as f:
    for class_idx, class_name in enumerate(container_verbs):
        f.write(f'{class_idx} {class_name}\n')
