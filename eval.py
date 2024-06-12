from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

y_true = [{"A", "B"}, {"A", "B", "C"}, {"A", "B", "D"}]
y_pred = [{"X", "B", "C"}, {"A", "C"}, {"B"}]

classes_set = {ii for i in [*y_true, *y_pred] for ii in i}
mlb = MultiLabelBinarizer(classes=[*classes_set])

y_true_binary = mlb.fit_transform(y_true)
y_pred_binary = mlb.transform(y_pred)

precision = precision_score(y_true_binary, y_pred_binary, average='micro')
recall = recall_score(y_true_binary, y_pred_binary, average='micro')
f1 = f1_score(y_true_binary, y_pred_binary, average='micro')

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
