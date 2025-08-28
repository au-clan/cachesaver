import gzip
import json

path = "datasets/dataset_hle_sample_without_images.jsonl.gz"
N = 5  # Number of samples to print

with gzip.open(path, 'rt', encoding='utf-8') as f:
    for i in range(N):
        line = f.readline()
        if not line:
            break  # Reached end of file
        sample = json.loads(line)
        print(f"\nSample {i+1}:")
        print(json.dumps(sample, indent=2))  # Nicely formatted
