from transformers import TrainingArguments
from datasets import load_metric
import torch
import numpy as np
from transformers import ViTForImageClassification
from transformers import ViTFeatureExtractor
from transformers import Trainer

from use_case import get_dataset_dict

model_name = 'google/vit-base-patch16-224-in21k'
fe = ViTFeatureExtractor.from_pretrained(model_name)
metric = load_metric("f1")


def collate_fn(batch):
  return {
    'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
    'labels': torch.tensor([x['labels'] for x in batch])
  }


def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1),
                        references=p.label_ids)


def transform(example_batch):
  inputs = fe([x for x in example_batch['image']], return_tensors='pt')
  inputs['labels'] = example_batch['labels']
  return inputs


ds = get_dataset_dict()
ds = ds.class_encode_column('labels')

labels = ds['train'].features['labels'].names

model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
    output_dir="../vit-test",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=False,  # changed
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,  # changed
    report_to='tensorboard',
    load_best_model_at_end=True,
)

prepared_ds = ds.with_transform(transform)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=fe,
)
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
