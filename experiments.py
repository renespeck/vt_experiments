from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from os import listdir

folder = 'data/ADE20K_experiments datasets/sub_train/positive/'

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

base = 'google/vit-base-patch16-224'
large = 'google/vit-large-patch16-224'
huge = 'google/vit-huge-patch14-224-in21k'

processor = ViTImageProcessor.from_pretrained(base)
model = ViTForImageClassification.from_pretrained(base)

print(len(model.config.id2label))
for i in range(0, len(model.config.id2label)):
  print(len(model.config.id2label[i]))

for images in os.listdir(folder):
  if images.endswith(".jpg"):
    print(images)
    image = Image.open(folder + images)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print(predicted_class_idx)
    print("Predicted class:", model.config.id2label[predicted_class_idx])
