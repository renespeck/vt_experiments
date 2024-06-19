import json

from eval import evaluation

save = 'results/sub_test_neg.json'

y_true, y_pred = [], []
with open(save) as f:
  for e in json.load(f):
    if str(e['scene'][-1]).__eq__("bathroom"):
      y_true.append("bathroom")
    else:
      y_true.append("no bathroom")
    if "p" in {*e['pre_labels']}:
      y_pred.append("bathroom")
    else:
      y_pred.append("no bathroom")

assert len(y_true) == len(y_pred)

evaluation(y_true, y_pred)
