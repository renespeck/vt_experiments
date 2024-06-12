import json

from eval import evaluation
from mapping import id2label_imagenet

save = 'results/sub_test_neg.json'


def class_mapper():
  ade20k_classes = {
    'shower',
    'cabin__outdoor',
    'bathhouse',
    'classroom',
    'seawall',
    'bathroom',
    'shower_room'
  }


def mapping(id: int):
  bathroom_ids = {435, 794, 861, 876, 896, 999}
  classroom_ids = {624}
  shower_ids = {}
  cabin__outdoor = {}
  bathhouse_id = {}
  shower_room_ids = {}
  bathhouse_ids = {}

  if id in bathroom_ids:
    return "bathroom"
  if id in classroom_ids:
    return "classroom"

  return id2label_imagenet[id]


y_true, y_pred = [], []
with open(save) as f:
  for e in json.load(f):
    y_true.append({e['scene'][-1]})
    p = mapping(int(e['pre_id']))
    y_pred.append({p})

assert len(y_true) == len(y_pred)
print(y_true)
print(y_pred)

evaluation(y_true, y_pred)
