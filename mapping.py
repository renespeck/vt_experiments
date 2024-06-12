from huggingface_hub import hf_hub_download
import json

# https://huggingface.co/datasets/huggingface/label-files/tree/main
repo_id = "huggingface/label-files"
file = 'imagenet-1k-id2label.json'
id2label = json.load(open(hf_hub_download(
    repo_id, file, repo_type="dataset"), "r")
)
id2label_imagenet = {int(k): v for k, v in id2label.items()}

repo_id = "huggingface/label-files"
file = 'ade20k-id2label.json'
id2label = json.load(open(hf_hub_download(
    repo_id, file, repo_type="dataset"), "r")
)
id2label_ade20k = {int(k): v for k, v in id2label.items()}
