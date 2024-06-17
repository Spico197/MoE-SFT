from packaging import version
from typing import Optional, List, Union, Any, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers.trainer import Trainer, logger
from transformers.utils import is_torch_tpu_available, is_sagemaker_mp_enabled
from transformers.integrations.deepspeed import deepspeed_init
from transformers.trainer_utils import (
    EvalLoopOutput,
    has_length,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_detach,
    IterableDatasetShard,
)


if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


class GateLoadRecordingTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,  # invalid arg here
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        gate_load = None
        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise NotImplementedError
                # raw_outputs = smp_forward_only(model, inputs)
                # if has_labels or loss_without_labels:
                #     if isinstance(raw_outputs, dict):
                #         loss_mb = raw_outputs["loss"]
                #         logits_mb = tuple(
                #             v
                #             for k, v in raw_outputs.items()
                #             if k not in ignore_keys + ["loss"]
                #         )
                #     else:
                #         loss_mb = raw_outputs[0]
                #         logits_mb = raw_outputs[1:]

                #     loss = loss_mb.reduce_mean().detach().cpu()
                #     logits = smp_nested_concat(logits_mb)
                # else:
                #     loss = None
                #     if isinstance(raw_outputs, dict):
                #         logits_mb = tuple(
                #             v for k, v in raw_outputs.items() if k not in ignore_keys
                #         )
                #     else:
                #         logits_mb = raw_outputs
                #     logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()
                    assert isinstance(outputs, dict)
                    if hasattr(outputs, "gate_load"):
                        # last layer
                        gate_load = outputs.gate_load
                        # (batch_size, num_experts)
                        gate_load = gate_load[-1]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                        if hasattr(outputs, "gate_load"):
                            # last layer
                            gate_load = outputs.gate_load
                            # (batch_size, num_experts)
                            gate_load = gate_load[-1]
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if gate_load is not None:
            gate_load = nested_detach(gate_load)

        return (loss, gate_load)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        gate_loads_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_gate_loads = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            if step >= args.max_eval_steps_per_type:
                break
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, gate_load = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            gate_load = gate_load.reshape(1, -1)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = (
                    losses
                    if losses_host is None
                    else nested_concat(losses_host, losses, padding_index=-100)
                )
            if gate_load is not None:
                gate_load = self.gather_function((gate_load))
                # (batch_size, num_experts)
                gate_loads_host = (
                    gate_load
                    if gate_loads_host is None
                    else torch.cat([gate_loads_host, gate_load])
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if gate_loads_host is not None:
                    gate_loads = nested_numpify(gate_loads_host)
                    all_gate_loads = (
                        gate_loads
                        if all_gate_loads is None
                        else np.concatenate((all_gate_loads, gate_loads), axis=0)
                    )

                # Set back to None to begin a new accumulation
                losses_host, gate_loads_host = None, None

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if gate_loads_host is not None:
            gate_loads = nested_numpify(gate_loads_host)
            all_gate_loads = (
                gate_loads
                if all_gate_loads is None
                else np.concatenate((all_gate_loads, gate_loads), axis=0)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        metrics = {}
        if all_gate_loads is not None:
            metrics["gate_load"] = all_gate_loads.mean(0).tolist()

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            metrics.update({"all_metrics": True})
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
