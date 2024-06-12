import json

from eval import evaluation

save = 'results/sub_train_pos.json'

y_true, y_pred = [], []
with open(save) as f:
  for e in json.load(f):
    y_true.append({e['scene'][-1]})
    y_pred.append({*e['pre_labels']})

assert len(y_true) == len(y_pred)
print(y_true)
print(y_pred)

evaluation(y_true, y_pred)
