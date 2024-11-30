import datasets
import os
from tqdm import tqdm
import json

# def load_instruction_data(file_path):
#     """Load instruction data from JSON file and convert to dataset format"""
#     with open(file_path, 'r') as f:
#         raw_data = json.load(f)
    
#     # Convert to format matching datasets structure
#     formatted_data = {
#         'train': [
#             {
#                 'text': f"Task: {item['instruction']}\nInput: {item['input']}\nCommand: {item['response']}\n\n"
#             }
#             for item in raw_data['data']
#         ]
#     }
#     return formatted_data

data = datasets.load_dataset("monology/pile-uncopyrighted")
print(data)

output_dir = "data/pile/"

instruction_data = load_instruction_data("instruction_data.json")

os.makedirs(output_dir, exist_ok=True)

max_num_per_shard = 1000000
ofid = 0
did = 0

for split in ["train"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    for d in tqdm(data[split]):
        f.write(json.dumps(d) + "\n")
        did += 1
        if did >= max_num_per_shard:
            f.close()
            ofid += 1
            did = 0
            f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    
    f.close()
