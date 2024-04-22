from generativeimage2text.model import get_git_model
from generativeimage2text.train import forward_backward
from transformers import BertTokenizer
import json

#Settings:
batch_size = 4

param = {}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = get_git_model(tokenizer, param)

with open("ocid_data/OCID-For-GIT/train.json", "r") as file:
    train_data_points = json.load(file)
    
sample_counter = 1
delimiter = "o_o"
scene_files = []
mask_files = []
captions = []
for key, value in train_data_points.items():
    scene_file_path = value.get("scene_file_path")
    mask_file_path = value.get("mask_file_path")
    scene_instance_id = value.get("scene_instance_id")
    caption = value.get("caption")
    
    scene_files.append(scene_file_path)
    mask_files.append(str(scene_instance_id) + delimiter + mask_file_path)
    captions.append(caption)
    
    if(sample_counter%batch_size == 0):
        forward_backward(model, scene_files, mask_files, captions)
        sample_counter = 0
        scene_files = []
        mask_files = []
        captions = []
    
    sample_counter += 1
    
if len(scene_files) != 0:
    forward_backward(model, scene_files, mask_files, captions)
    sample_counter = 0
    scene_files = []
    mask_files = []
    captions = []