import os

from datasets import Dataset, DatasetDict
from pathlib import Path
from PIL import Image


def gen_train():
  for data in get_data('../data/ADE20K/sub_train/negative/', "n"):
    yield data
  for data in get_data('../data/ADE20K/sub_train/positive/', "p"):
    yield data


def gen_test():
  for data in get_data('../data/ADE20K/sub_test/negative/', "n"):
    yield data
  for data in get_data('../data/ADE20K/sub_test/positive/', "p"):
    yield data


def get_data(folder, label):
  data = []
  # read data
  for js_file in os.listdir(folder):
    if js_file.endswith(".json"):
      img_path = f'{folder}{Path(js_file).stem}.jpg'
      if os.path.isfile(img_path):
        # json and jpg file exists
        image_read = Image.open(img_path)
        data.append({"image": image_read, 'labels': label})
  return data


def get_dataset_dict():
  train = Dataset.from_generator(gen_train)
  test = Dataset.from_generator(gen_test)
  return DatasetDict({"train": train, "test": test})
