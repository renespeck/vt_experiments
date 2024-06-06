from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from os import listdir
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import json

folder = 'data/ADE20K/sub_test/positive/'

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

base = 'google/vit-base-patch16-224'
large = 'google/vit-large-patch16-224'
huge = 'google/vit-huge-patch14-224-in21k'

processor = ViTImageProcessor.from_pretrained(large)
model = ViTForImageClassification.from_pretrained(large)

data = []

images = os.listdir(folder)
# images = images[0:10]
for image in images:
  if image.endswith(".jpg"):
    image_read = Image.open(folder + image)

    inputs = processor(images=image_read, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    pre = model.config.id2label[predicted_class_idx]

    data.append({
      'id': f'{image}',
      'pre': [*pre.split(', ')]
    })

with open('sub_test_pos.json', 'w') as f:
  json.dump(data, f, indent=4, sort_keys=True)
