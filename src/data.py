import random
from pathlib import Path
from typing import Iterator, Any, Mapping, Dict

import torch
import transformers
import numpy as np
from loguru import logger
from torch.utils.data import IterableDataset, Dataset
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother

from src.utils.io import load_jsonlines_iter, load_jsonlines
from src.utils.conversation import Conversation

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    instances,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    tokenizer_legacy = getattr(tokenizer, "legacy", True)
    conv = Conversation()
    conv.sep2 = tokenizer.eos_token
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, ins in enumerate(instances):
        if roles[ins["conversations"][0]["from"]] != roles["human"]:
            # Skip the first one if it is not from human
            ins["conversations"] = ins["conversations"][1:]

        conv.clear_msg()
        sys_msg = ins.get("system_prompt")
        if sys_msg is not None:
            conv.set_system_message(sys_msg)
        else:
            conv.set_system_message("")
        for j, turn in enumerate(ins["conversations"]):
            role = roles[turn["from"]]
            assert role == conv.roles[j % 2], f"{i}/{j}"
            conv.append_message(role, turn["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    res = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = res["input_ids"]
    attention_masks = res["attention_mask"]
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    # attention_masks = torch.ones_like(input_ids)
    for conversation, target, attention_mask in zip(conversations, targets, attention_masks):
        turns = conversation.split(conv.sep2)
        # the eos token is included in `total_len`, llama2 will add bos token
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + len(turns) * int(tokenizer.pad_token == tokenizer.eos_token)
        # attention_mask[total_len:] = 0
        total_len = attention_mask.sum()

        cur_len = 0
        has_bos = False
        if target[0] == tokenizer.bos_token_id:
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID  # bos token
            has_bos = True
        for i, turn in enumerate(turns):
            if turn == "":
                break
            # +1: add sep2 token
            turn_len = len(tokenizer(turn).input_ids) - int(has_bos) + 1

            # sep: " ASSISTANT: "
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct: bos and the last space token
            # -1 means remove extra suffix space in sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - int(has_bos) - 1

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
            # if i < len(turns) - 1:
            #     # plus one for sep2 token (eos)
            #     cur_len += 1

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logger.info(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks,
    )


# def preprocess(
#     instances,
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     tokenizer_legacy = getattr(tokenizer, "legacy", None)
#     conv = Conversation()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, ins in enumerate(instances):
#         if roles[ins["conversations"][0]["from"]] != roles["human"]:
#             # Skip the first one if it is not from human
#             ins["conversations"] = ins["conversations"][1:]

#         conv = Conversation()
#         sys_msg = ins.get("system_prompt")
#         if sys_msg is not None:
#             conv.set_system_message(sys_msg)
#         else:
#             conv.set_system_message("")
#         for j, turn in enumerate(ins["conversations"]):
#             role = roles[turn["from"]]
#             assert role == conv.roles[j % 2], f"{i}/{j}"
#             conv.append_message(role, turn["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations
#     input_ids = tokenizer(
#         conversations,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=tokenizer.model_max_length,
#         truncation=True,
#     ).input_ids
#     targets = input_ids.clone()

#     # Mask targets. Only compute loss on the assistant outputs.
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         turns = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_TOKEN_ID
#         for i, turn in enumerate(turns):
#             if turn == "":
#                 break
#             turn_len = len(tokenizer(turn).input_ids)

#             parts = turn.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep
#             # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
#             instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             if i != 0 and not tokenizer_legacy:
#                 # The legacy and non-legacy modes handle special tokens differently
#                 instruction_len -= 1

#             # Ignore the user instructions
#             target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
#             cur_len += turn_len

#             if i != 0 and not tokenizer_legacy:
#                 # The legacy and non-legacy modes handle special tokens differently
#                 cur_len -= 1

#         target[cur_len:] = IGNORE_TOKEN_ID

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_TOKEN_ID
#                 logger.info(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" #turn = {len(turns) - 1}. (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )


def get_uniform_sampling_ratio(folder: str) -> dict:
    sub_folders = [p for p in Path(folder).glob("*") if p.is_dir()]
    sampling_ratio = 1.0 / len(sub_folders)

    sampling_map = {subfolder.name: sampling_ratio for subfolder in sub_folders}
    return sampling_map


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        try:
            features = [vars(f) for f in features]
        except TypeError:
            print(len(features), type(features[0]), features[0])
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


class CachedJsonlDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        tokenizer: PreTrainedTokenizer,
        seed: int = 1227,
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.data = load_jsonlines(datapath)
        self.rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        ins = self.data[index]
        processed = preprocess([ins], self.tokenizer)
        ins = {}
        for key in processed:
            ins[key] = processed[key][0]
        return ins

    def state_dict(self):
        return {
            "datapath": self.datapath,
            "seed": self.seed,
            "rng": self.rng.getstate(),
        }


class PackedJsonlDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        seed: int = 1227,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer

        data_dir_path = Path(data_dir)
        filepaths = sorted(data_dir_path.glob("**/*.jsonl"))
        self.rng.shuffle(filepaths)
        self.filepaths = filepaths
        self.curr_filepath = None

    def __iter__(self) -> Iterator:
        while True:
            for filepath in self.filepaths:
                self.curr_filepath = filepath
                logger.debug(f"Iter over jsonl file: {filepath}")
                for ins in load_jsonlines_iter(filepath):
                    processed = preprocess([ins], self.tokenizer)
                    ins = {}
                    for key in processed:
                        ins[key] = processed[key][0]
                    yield ins

    def state_dict(self):
        return {
            "data_dir": self.data_dir,
            "seed": self.seed,
            "rng": self.rng.getstate(),
            "filepaths": self.filepaths,
            "curr_filepath": self.curr_filepath,
        }


class SubDirWeightedPackedJsonlDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: PreTrainedTokenizer,
        prob_map: dict[str, float] | list[tuple[str, int]] = None,
        seed: int = 1227,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.dataset_dir_path = Path(dataset_dir)
        data_types = [p.stem for p in self.dataset_dir_path.glob("*") if p.is_dir()]
        if prob_map is None:
            prob_map = {str(data_type): 1.0 for data_type in data_types}
        for data_type in data_types:
            assert data_type in prob_map
        for data_type in prob_map:
            if data_type not in data_types:
                logger.warning(
                    f"Task type {data_type} not found in dataset dir. Skip it."
                )

        self.source2idx = {}
        self.prob_map = {}
        if isinstance(prob_map, dict):
            _prob_map = list(prob_map.items())
        elif isinstance(prob_map, list):
            _prob_map = prob_map
        else:
            raise ValueError(f"Unknown prob_map type: {type(prob_map)}")
        for data_type, sampling_weight in _prob_map:
            self.source2idx[data_type] = len(self.source2idx)
            self.prob_map[data_type] = sampling_weight

        logger.info(f"Prob map: {self.prob_map}")
        self.data_type_to_dataset = {}
        for data_type in data_types:
            ds = iter(
                PackedJsonlDataset(
                    str(self.dataset_dir_path.joinpath(data_type)),
                    tokenizer,
                    seed=seed,
                )
            )
            self.data_type_to_dataset[data_type] = ds

    def update_prob_map(self, new_prob_map: dict):
        logger.info(f"Update existed prob map: {new_prob_map}, old: {self.prob_map}")
        self.prob_map.update(new_prob_map)

    def update_existed_prob_map(self, new_prob_map: dict):
        logger.info(f"Update existed prob map: {new_prob_map}, old: {self.prob_map}")
        for name in self.prob_map:
            if name in new_prob_map:
                self.prob_map[name] = new_prob_map[name]

    def __iter__(self) -> Iterator:
        while len(self.data_type_to_dataset) > 0:
            candidate_data_types = list(self.data_type_to_dataset.keys())
            weights = [self.prob_map[data_type] for data_type in candidate_data_types]
            choice = self.rng.choices(candidate_data_types, weights=weights, k=1)[0]
            try:
                yield next(self.data_type_to_dataset[choice])
            except StopIteration:
                return
            except KeyError as err:
                print(choice)
                raise err


def get_cached_datasets_from_dir(
    folder: str, tokenizer: PreTrainedTokenizer, seed: int = 1227
):
    fd = Path(folder)
    files = list(fd.glob("*.jsonl"))
    name2ds = {}
    for file in files:
        name = file.stem
        ds = CachedJsonlDataset(str(file), tokenizer, seed=seed)
        name2ds[name] = ds
    return name2ds
