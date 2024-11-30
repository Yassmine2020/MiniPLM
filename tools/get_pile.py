import datasets
import os
from tqdm import tqdm
import json

# Load the dataset
data = datasets.load_dataset("monology/pile-uncopyrighted")
print("Original dataset:", data)

# Create a smaller version using select
num_examples = len(data['train'])
reduced_size = num_examples // 100  # Take 1/10th of the data
indices = range(0, num_examples, 100)  # Take every 10th example
reduced_data = data['train'].select(indices)
print(f"Reduced dataset size: {len(reduced_data)} (originally {num_examples})")

output_dir = "data/pile/"
os.makedirs(output_dir, exist_ok=True)

max_num_per_shard = 1000000
ofid = 0
did = 0

for split in ["train"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    for d in tqdm(reduced_data):  # Using reduced_data instead of data[split]
        f.write(json.dumps(d) + "\n")
        did += 1
        if did >= max_num_per_shard:
            f.close()
            ofid += 1
            did = 0
            f = open(os.path.join(output_dir, split, f"{ofid}.jsonl"), "w")
    
    f.close()

print("Processing completed.")
