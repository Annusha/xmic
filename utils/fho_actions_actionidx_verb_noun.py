import json
from collections import defaultdict


egopath = '/mnt/graphics_ssd/nimble/users/annakukleva/data/ego4d/annotations/fho_lta_val.json'
egopath = '/mnt/graphics_ssd/nimble/users/annakukleva/data/ego4d/annotations/fho_lta_train.json'



with open(egopath, 'r') as f:
    annotations = json.load(f)
    annotations = annotations['clips']

actions_val = defaultdict(int)
for item in annotations:
    actions_val[(item['noun'], item['verb'])] += 1

actions = set()
for item in actions_train.keys():
    actions.add(item)
for item in actions_val.keys():
    actions.add(item)


map_path = '/mnt/graphics_ssd/nimble/users/annakukleva/data/ego4d/annotations/fho_actions_actionidx_verb_noun.txt'

with open(map_path, 'w') as f:
    for idx, item in enumerate(actions):
        a = f.write(f'{idx} {item[1]} {item[0]}\n')
