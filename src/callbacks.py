import os
import re
from pathlib import Path
from typing import Literal, Dict, Optional

import torch
import numpy as np
from loguru import logger
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.utils import is_flash_attn_2_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.utils.config import TrainingArguments
from src.utils.io import append_jsonlines
from src.utils import vis as vis


ADAPTER_MODEL_DIRNAME = "adapter_model"
SAMPLING_INFO_DIRNAME = "sampling_info"
SAMPLING_DATA_FILENAME = "data.jsonl"


class AdaptiveSamplingCallback(TrainerCallback):
    def __init__(
        self,
        criterion: Optional[Literal["min", "max", "mean"]] = "mean",
        sim_type: Optional[Literal["cos", "l2", "~l2"]] = "cos",
        eta: float = 10.0,
        c: float = 5e-2,
    ):
        assert is_flash_attn_2_available(), "Make sure you have flash-attn installed"
        self.criterion = criterion
        self.sim_type = sim_type
        self.eta = eta
        self.c = c
        self.prob_map = {}
        self.output_dir = None
        self.data_path = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        train_dataloader = kwargs["train_dataloader"]
        self.prob_map = train_dataloader.dataset.prob_map
        self.output_dir = Path(args.output_dir).joinpath(SAMPLING_INFO_DIRNAME)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.data_path = self.output_dir.joinpath(SAMPLING_DATA_FILENAME)

    def _sim(self, loads: np.ndarray) -> tuple:
        def _agg(arr):
            # compute the deviation based on a given criterion
            if self.criterion == "min":
                arr_v = arr.min(axis=1)
            elif self.criterion == "max":
                arr_v = arr.max(axis=1)
            elif self.criterion == "mean":
                arr_v = arr.mean(axis=1)
            else:
                raise ValueError(f"Invalid criterion: {self.criterion}")
            return arr_v

        if self.sim_type == "cos":
            # cos sim, target: (num_data_types, num_data_types)
            norm = np.linalg.norm(loads, axis=1, keepdims=True)
            normalized = loads / norm
            sim = np.dot(normalized, normalized.T)
            delta = 1.0 - _agg(sim)
        elif self.sim_type == "l2":
            sim = np.dot(loads, loads.T)
            delta = np.linalg.norm(loads[:, np.newaxis] - loads, axis=2)
            delta = _agg(delta)
        elif self.sim_type == "~l2":
            sim = -np.dot(loads, loads.T)
            delta = np.linalg.norm(loads[:, np.newaxis] - loads, axis=2)
            delta = _agg(delta)
            delta_ind = np.argsort(delta)
            # swap the min scores with the max scores, inverse the delta
            delta[delta_ind] = delta[delta_ind[::-1]]
        else:
            raise ValueError(f"Invalid sim type: {self.sim_type}")

        return sim, delta

    def _update_weights(
        self,
        ori_weights: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        def _softmax(vec: np.ndarray) -> np.ndarray:
            exps = np.exp(vec - np.max(vec))
            return exps / exps.sum()

        alpha = np.log(ori_weights) + self.eta * delta
        # softmax on alpha
        alpha = _softmax(alpha)
        # re-normalize and smooth domain weights
        new_weights = (1 - self.c) * alpha + self.c / len(alpha)
        new_weights = new_weights / new_weights.sum()
        return new_weights

    def _update_prob_map(self, name2load: Dict[str, float]) -> dict:
        # sanity check
        new_names = set(name2load.keys())
        old_names = set(self.prob_map.keys())
        assert len(new_names) == len(
            old_names
        ), f"New names {new_names} do not match old names {old_names}"
        assert len(new_names & old_names) == len(
            new_names
        ), f"New names {new_names} do not match old names {old_names}"
        # update
        names = sorted(new_names)
        loads = [name2load[name] for name in names]
        # (num_data_types, num_experts)
        loads = np.array(loads)
        # sim: (num_data_types, num_data_types)
        # delta: (num_data_types,)
        sim, delta = self._sim(loads)
        # update the probability map
        ori_weights = np.array([self.prob_map[name] for name in names])
        new_weights = self._update_weights(ori_weights, delta)
        new_prob_map = {name: weight for name, weight in zip(names, new_weights)}
        return new_prob_map, sim

    def _parse_name2load(self, metrics: dict) -> dict:
        name2load = {}
        key = re.compile(r"eval_(.*?)_gate_load")
        for name, val in metrics.items():
            if key.match(name):
                obj = key.search(name)
                data_type = obj.group(1)
                # val: np.ndarray
                val = np.array(val)
                _load = val / val.sum()
                name2load[data_type] = _load.tolist()
        return name2load

    @torch.inference_mode()
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        if not metrics.get("all_metrics", False):
            return

        train_dataloader = kwargs["train_dataloader"]
        if not hasattr(train_dataloader.dataset, "update_existed_prob_map"):
            return

        name2load = self._parse_name2load(metrics)
        new_prob_map, sim = self._update_prob_map(name2load)
        if state.is_world_process_zero:
            logger.info(f"Update prob map - old: {self.prob_map}, new: {new_prob_map}")
            append_jsonlines(
                [
                    {
                        "step": state.global_step,
                        "old_prob_map": self.prob_map,
                        "new_prob_map": new_prob_map,
                        "sim": sim.tolist(),
                        "name2load": name2load,
                    }
                ],
                self.data_path,
            )
            step_info_dir = self.output_dir.joinpath(f"{state.global_step}")
            step_info_dir.mkdir(exist_ok=True, parents=True)
            # plot load
            names = sorted(name2load.keys())
            loads = np.array([name2load[name] for name in names])
            vis.heatmap(
                loads,
                title="Gate Load",
                ylabels=names,
                filepath=step_info_dir.joinpath("load.pdf"),
            )
            # plot sim
            vis.heatmap(
                sim,
                title="Gate Selection Similarity",
                xlabels=names,
                ylabels=names,
                filepath=step_info_dir.joinpath("sim.pdf"),
            )
            # plot old and new prob map
            labels = ["old", "new"]
            old_new_plot_data = {
                name: [self.prob_map[name], new_prob_map[name]] for name in names
            }
            vis.group_bar(
                old_new_plot_data,
                labels,
                title="Sampling Weights",
                filepath=step_info_dir.joinpath("prob_map.pdf"),
            )
            logger.info(f"Save step info to {str(step_info_dir)}")
        self.prob_map = new_prob_map
        train_dataloader.dataset.update_existed_prob_map(new_prob_map)


class RefLossSamplingCallback(TrainerCallback):
    def __init__(
        self,
        name2ref_loss: Dict[str, float],
        eta: float = 1.0,
        c: float = 1e-4,
    ):
        assert is_flash_attn_2_available(), "Make sure you have flash-attn installed"
        self.name2ref_loss = name2ref_loss
        self.eta = eta
        self.c = c
        self.prob_map = {}
        self.output_dir = None
        self.data_path = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        train_dataloader = kwargs["train_dataloader"]
        self.prob_map = train_dataloader.dataset.prob_map
        self.output_dir = Path(args.output_dir).joinpath(SAMPLING_INFO_DIRNAME)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.data_path = self.output_dir.joinpath(SAMPLING_DATA_FILENAME)

    def _update_weights(
        self,
        ori_weights: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        def _softmax(vec: np.ndarray) -> np.ndarray:
            exps = np.exp(vec - np.max(vec))
            return exps / exps.sum()

        alpha = np.log(ori_weights) + self.eta * delta
        # softmax on alpha
        alpha = _softmax(alpha)
        # re-normalize and smooth domain weights
        new_weights = (1 - self.c) * alpha + self.c / len(alpha)
        new_weights = new_weights / new_weights.sum()
        return new_weights

    def _update_prob_map(self, name2loss: Dict[str, float]) -> dict:
        # sanity check
        new_names = set(name2loss.keys())
        old_names = set(self.prob_map.keys())
        assert len(new_names) == len(
            old_names
        ), f"New names {new_names} do not match old names {old_names}"
        assert len(new_names & old_names) == len(
            new_names
        ), f"New names {new_names} do not match old names {old_names}"
        # update
        names = sorted(new_names)
        delta = np.array([name2loss[name] - self.name2ref_loss[name] for name in names])
        # update the probability map
        ori_weights = np.array([self.prob_map[name] for name in names])
        new_weights = self._update_weights(ori_weights, delta)
        new_prob_map = {name: weight for name, weight in zip(names, new_weights)}
        return new_prob_map, delta

    def _parse_name2loss(self, metrics: dict) -> dict:
        name2loss = {}
        key = re.compile(r"eval_(.*?)_loss")
        for name, val in metrics.items():
            if key.match(name):
                obj = key.search(name)
                data_type = obj.group(1)
                name2loss[data_type] = float(val)
        return name2loss

    @torch.inference_mode()
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        if not metrics.get("all_metrics", False):
            return

        train_dataloader = kwargs["train_dataloader"]
        if not hasattr(train_dataloader.dataset, "update_existed_prob_map"):
            return

        name2loss = self._parse_name2loss(metrics)
        new_prob_map, delta = self._update_prob_map(name2loss)
        if state.is_world_process_zero:
            logger.info(f"Update prob map - old: {self.prob_map}, new: {new_prob_map}")
            append_jsonlines(
                [
                    {
                        "step": state.global_step,
                        "old_prob_map": self.prob_map,
                        "new_prob_map": new_prob_map,
                        "delta": delta.tolist(),
                        "name2loss": name2loss,
                        "name2ref_loss": self.name2ref_loss,
                    }
                ],
                self.data_path,
            )
            step_info_dir = self.output_dir.joinpath(f"{state.global_step}")
            step_info_dir.mkdir(exist_ok=True, parents=True)
            # plot load
            names = sorted(name2loss.keys())
            # plot old and new prob map
            labels = ["old", "new"]
            old_new_plot_data = {
                name: [self.prob_map[name], new_prob_map[name]] for name in names
            }
            vis.group_bar(
                old_new_plot_data,
                labels,
                title="Sampling Weights",
                filepath=step_info_dir.joinpath("prob_map.pdf"),
            )
            name2delta = {name: delta for name, delta in zip(names, delta)}
            vis.bar(
                name2delta,
                title="Differences to Reference Loss",
                filepath=step_info_dir.joinpath("loss_delta.pdf"),
            )
            logger.info(f"Save step info to {str(step_info_dir)}")
        self.prob_map = new_prob_map
        train_dataloader.dataset.update_existed_prob_map(new_prob_map)


class RandomSamplingCallback(TrainerCallback):
    def __init__(self, digit=4, **kwargs):
        self.digit = digit
        self.prob_map = {}
        self.output_dir = None
        self.data_path = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        train_dataloader = kwargs["train_dataloader"]
        self.prob_map = train_dataloader.dataset.prob_map
        self.output_dir = Path(args.output_dir).joinpath(SAMPLING_INFO_DIRNAME)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.data_path = self.output_dir.joinpath(SAMPLING_DATA_FILENAME)

    def _update_weights(
        self,
        ori_weights: np.ndarray,
    ) -> np.ndarray:
        num_datasets = len(ori_weights)
        new_weights = np.random.random(num_datasets)
        new_weights = new_weights / new_weights.sum()
        # cnt = np.random.randint(0, num_datasets - 1, (10**self.digit,))
        # bincnt = np.bincount(cnt, minlength=num_datasets)
        # new_weights = bincnt / bincnt.sum()
        return new_weights

    def _update_prob_map(self, name2load: Dict[str, float]) -> dict:
        # sanity check
        new_names = set(name2load.keys())
        old_names = set(self.prob_map.keys())
        assert len(new_names) == len(
            old_names
        ), f"New names {new_names} do not match old names {old_names}"
        assert len(new_names & old_names) == len(
            new_names
        ), f"New names {new_names} do not match old names {old_names}"
        # update
        names = sorted(new_names)
        loads = [name2load[name] for name in names]
        # (num_data_types, num_experts)
        loads = np.array(loads)
        # update the probability map
        ori_weights = np.array([self.prob_map[name] for name in names])
        new_weights = self._update_weights(ori_weights)
        new_prob_map = {name: weight for name, weight in zip(names, new_weights)}
        return new_prob_map, np.random.rand(len(names), len(names))

    def _parse_name2load(self, metrics: dict) -> dict:
        name2load = {}
        key = re.compile(r"eval_(.*?)_gate_load")
        for name, val in metrics.items():
            if key.match(name):
                obj = key.search(name)
                data_type = obj.group(1)
                # val: np.ndarray
                val = np.array(val)
                _load = val / val.sum()
                name2load[data_type] = _load.tolist()
        return name2load

    @torch.inference_mode()
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        if not metrics.get("all_metrics", False):
            return

        train_dataloader = kwargs["train_dataloader"]
        if not hasattr(train_dataloader.dataset, "update_existed_prob_map"):
            return

        name2load = self._parse_name2load(metrics)
        new_prob_map, _ = self._update_prob_map(name2load)
        if state.is_world_process_zero:
            logger.info(f"Update prob map - old: {self.prob_map}, new: {new_prob_map}")
            append_jsonlines(
                [
                    {
                        "step": state.global_step,
                        "old_prob_map": self.prob_map,
                        "new_prob_map": new_prob_map,
                        "name2load": name2load,
                    }
                ],
                self.data_path,
            )
            step_info_dir = self.output_dir.joinpath(f"{state.global_step}")
            step_info_dir.mkdir(exist_ok=True, parents=True)
            # plot load
            names = sorted(name2load.keys())
            loads = np.array([name2load[name] for name in names])
            vis.heatmap(
                loads,
                title="Gate Load",
                ylabels=names,
                filepath=step_info_dir.joinpath("load.pdf"),
            )
            # plot old and new prob map
            labels = ["old", "new"]
            old_new_plot_data = {
                name: [self.prob_map[name], new_prob_map[name]] for name in names
            }
            vis.group_bar(
                old_new_plot_data,
                labels,
                title="Sampling Weights",
                filepath=step_info_dir.joinpath("prob_map.pdf"),
            )
            logger.info(f"Save step info to {str(step_info_dir)}")
        self.prob_map = new_prob_map
        train_dataloader.dataset.update_existed_prob_map(new_prob_map)


class RandomBatchSamplingCallback(RandomSamplingCallback):
    def _update_weights(
        self,
        ori_weights: np.ndarray,
    ) -> np.ndarray:
        new_weights = 1e-5 * np.ones(len(ori_weights))
        random_choice = np.random.choice(len(ori_weights))
        new_weights[random_choice] = 1.0
        new_weights /= new_weights.sum()
        return new_weights


class CycleSamplingCallback(RandomSamplingCallback):
    def _update_weights(
        self,
        ori_weights: np.ndarray,
    ) -> np.ndarray:
        new_weights = np.roll(ori_weights, 1)
        return new_weights


class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, ADAPTER_MODEL_DIRNAME)
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, ADAPTER_MODEL_DIRNAME)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        self.save_model(args, state, kwargs)
        touch(os.path.join(args.output_dir, 'completed'))
