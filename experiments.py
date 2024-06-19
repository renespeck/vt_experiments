from pathlib import Path

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
import json

folder = 'data/ADE20K/sub_train/negative/'
save = 'results/sub_train_neg.json'

base = 'google/vit-base-patch16-224'
large = 'vit-test'  # 'google/vit-large-patch16-224'
huge = 'google/vit-huge-patch14-224-in21k'

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
processor = ViTImageProcessor.from_pretrained(large)
model = ViTForImageClassification.from_pretrained(large)

data = []
images = os.listdir(folder)
for image in images:
  if image.endswith(".jpg"):
    js_file = Path(image).stem
    js_path = f'{folder}{js_file}.json'
    if os.path.isfile(js_path):

      with open(js_path) as f:
        d = json.load(f)

      image_read = Image.open(folder + image)
      inputs = processor(images=image_read, return_tensors="pt")
      outputs = model(**inputs)
      logits = outputs.logits

      predicted_class_idx = logits.argmax(-1).item()
      pre = model.config.id2label[predicted_class_idx]

      objects = set(o['name'] for o in d['annotation']['object'])
      objects_set = set()
      for o in objects:
        for i in o.split(', '):
          objects_set.add(i)

      data.append({
        'id': f'{image}',
        'folder': d['annotation']['folder'],
        'scene': d['annotation']['scene'],
        'objects': [*objects_set],
        'pre_id': f'{predicted_class_idx}',
        'pre_labels': [*pre.split(', ')]
      })
    else:
      print(f'FileNotFoundError: {js_path}')

with open(save, 'w') as f:
  json.dump(data, f, indent=4, sort_keys=True)
