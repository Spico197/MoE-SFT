"""
mkdir -p data/open_orca_clustered/1
cp data/open_orca/merged_samples_-1/fschat_0.jsonl data/open_orca_clustered/1
srun -p MoE -n 1 -N 1 --gres gpu:1 --mem 128G python -m src.core.clustering --do_emb --do_train --do_eval -n 2 -m data/open_orca_clustered/2 -o data/open_orca_clustered/2
mkdir -p data/open_orca_clustered/4
mkdir -p data/open_orca_clustered/8
mkdir -p data/open_orca_clustered/16
cp data/open_orca_clustered/2/emb.npy data/open_orca_clustered/4/emb.npy
cp data/open_orca_clustered/2/emb.npy data/open_orca_clustered/8/emb.npy
cp data/open_orca_clustered/2/emb.npy data/open_orca_clustered/16/emb.npy
srun -p MoE -n 1 -N 1 --mem 128G python -m src.core.clustering --do_train --do_eval -n 4 -m data/open_orca_clustered/4 -o data/open_orca_clustered/4
srun -p MoE -n 1 -N 1 --mem 128G python -m src.core.clustering --do_train --do_eval -n 8 -m data/open_orca_clustered/8 -o data/open_orca_clustered/8
srun -p MoE -n 1 -N 1 --mem 128G python -m src.core.clustering --do_train --do_eval -n 16 -m data/open_orca_clustered/16 -o data/open_orca_clustered/16
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from loguru import logger
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from src.utils.io import load_jsonlines_iter
from src.utils.conversation import Conversation


CLUSTERING_MODEL_NAME = "clustering.model"
EMBEDDING_NAME = "emb.npy"


class TextClustering:
    def __init__(
        self, num_clusters: int = 16, encoder: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.kmeans = KMeans(n_clusters=num_clusters, verbose=True)
        self.emb = SentenceTransformer(encoder)

    @property
    def num_clusters(self) -> int:
        return self.kmeans.n_clusters

    def encode_emb(
        self, sentences: list[str], show_progress_bar: bool = False
    ) -> np.ndarray:
        arr: np.ndarray = self.emb.encode(
            sentences=sentences, show_progress_bar=show_progress_bar
        )
        return arr

    def fit_emb(self, emb: np.ndarray):
        self.kmeans.fit(emb)

    def fit(self, sentences: list[str]):
        emb_arr = self.encode_emb(sentences)
        self.kmeans.fit(emb_arr)

    def predict_emb(self, emb: np.ndarray) -> list[int]:
        return self.kmeans.predict(emb).tolist()

    def predict(self, sentences: list[str]) -> list[int]:
        emb_arr = self.encode_emb(sentences)
        return self.predict_emb(emb_arr)

    def save_pretrained(self, folder: str):
        model_path = Path(folder) / CLUSTERING_MODEL_NAME
        model_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.kmeans, model_path)

    @classmethod
    def from_pretrained(cls, folder: str):
        model_path = Path(folder) / CLUSTERING_MODEL_NAME
        kmeans = joblib.load(model_path)
        model = cls()
        model.kmeans = kmeans
        return model


def main(args):
    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    contents = []
    content2ins = {}

    if args.do_emb:
        logger.info("Loading contents")
        for ins in load_jsonlines_iter(args.filepath):
            string = Conversation.parse(ins)
            contents.append(string)
            content2ins[string] = ins
        model = TextClustering(num_clusters=args.num_clusters)
        emb_arr = model.encode_emb(contents, show_progress_bar=True)
        emb_filepath = str(out_p / EMBEDDING_NAME)
        logger.info(f"Dumping emb to {emb_filepath}")
        np.save(emb_filepath, emb_arr)

    if args.do_train:
        model = TextClustering(num_clusters=args.num_clusters)
        logger.info("Fitting model")
        emb_filepath = out_p.joinpath(EMBEDDING_NAME)
        if emb_filepath.exists():
            logger.info(f"Loading emb from {str(emb_filepath)}")
            emb_arr = np.load(emb_filepath)
            model.fit_emb(emb_arr)
        elif len(contents) > 0:
            model.fit(contents)
        else:
            raise RuntimeError("emb file does not exist and contents are empty")
        logger.info("Saving model")
        model.save_pretrained(args.model_dir)

    if args.do_eval:
        logger.info("Loading model")
        model = TextClustering.from_pretrained(args.model_dir)
        logger.info("Loading contents")
        num_tot = len(content2ins)
        if num_tot <= 0:
            for ins in load_jsonlines_iter(args.filepath):
                string = Conversation.parse(ins)
                content2ins[string] = ins
                num_tot += 1

        labels = []
        bsz = 32
        batch = []
        bar = tqdm(total=num_tot)
        logger.info(f"file: {args.filepath}")
        contents = list(content2ins.keys())
        for content in tqdm(contents, desc="Predicting"):
            if len(batch) == bsz:
                preds = model.predict(batch)
                labels.extend(preds)
                bar.update(len(preds))
                batch.clear()
            else:
                batch.append(content)

        if len(batch) > 0:
            preds = model.predict(batch)
            labels.extend(preds)
            bar.update(len(preds))
            batch.clear()
        logger.info("Predicting finished")

        logger.info("Dumping results")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        cluster_to_num = defaultdict(lambda: 0)
        cluster_to_fout = {
            i: open(out_dir / f"fschat_{i}.jsonl", "w") for i in range(args.num_clusters)
        }
        for content, label in zip(contents, labels):
            ins = content2ins[content]
            cluster_to_fout[label].write(f"{json.dumps(ins, ensure_ascii=False)}\n")
            cluster_to_num[label] += 1

        for fp in cluster_to_fout.values():
            fp.close()

        logger.info(f"Done: {dict(cluster_to_num)}")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["LOGLEVEL"] = "INFO"

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_emb", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        default="data/open_orca/merged_samples_-1/fschat_0.jsonl",
    )
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-n", "--num_clusters", type=int, default=8)
    parser.add_argument("-m", "--model_dir", type=str)
    args = parser.parse_args()

    main(args)
