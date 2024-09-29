import random
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk

from src.utils.io import load_jsonlines_iter

# ds = load_dataset("cais/mmlu", "all", split="test")
# new_ds = ds.shuffle(seed=1227).select(range(1000))
# new_ds.save_to_disk("data/cais_mmlu_1000")

mmlu_texts = []
ds = load_from_disk("data/cais_mmlu_1000")
for ins in ds:
    mmlu_texts.append(ins["question"])


gate_load_data_dir = Path("data/merged_splits_gate_load_data_aggregated")


interval_to_text = defaultdict(list)
for datapath in gate_load_data_dir.glob("*.jsonl"):
    interval = datapath.stem.split("_")
    interval = (float(interval[0]), float(interval[1]))
    for ins in load_jsonlines_iter(datapath):
        interval_to_text[interval].append(ins["conversations"][0]["value"])
intervals = sorted(list(interval_to_text.keys()))


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Calculate embeddings by calling model.encode()
mmlu_embeddings = model.encode(mmlu_texts)
print(mmlu_embeddings.shape)

rand = random.Random(1227)
for interval in intervals:
    texts = rand.choices(interval_to_text[interval], k=1000)
    interval_embeddings = model.encode(texts)
    print(interval_embeddings.shape)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(mmlu_embeddings, interval_embeddings)
    print(interval, similarities.mean())

"""
(0.0, 0.0125) tensor(0.0402)  34.18
(1000, 384)
(0.0125, 0.025) tensor(0.0391)  37.12
(1000, 384)
(0.025, 0.0375) tensor(0.0435)  38.19
(1000, 384)
(0.0375, 0.05) tensor(0.0471)  40.04
(1000, 384)
(0.05, 0.0625) tensor(0.0515)  37.38
(1000, 384)
(0.0625, 0.075) tensor(0.0519)  38.10
(1000, 384)
(0.075, 0.0875) tensor(0.0512)  36.89
(1000, 384)
(0.0875, 0.1) tensor(0.0523)  35.83
(1000, 384)
"""
