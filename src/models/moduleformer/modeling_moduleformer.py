""" PyTorch ModuleFormer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.nn import functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

from transformers.activations import get_activation
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_moduleformer import ModuleFormerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "moduleformer-small"
_CONFIG_FOR_DOC = "ModuleFormerConfig"


# SPARSEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "moduleformer-small",
#     # See all ModuleFormer models at https://huggingface.co/models?filter=moduleformer
# ]


# @torch.jit.script
def log_gmm_posterior(z, expert_centroids):
    """
    Compute the log posterior probabilities of data points z belonging to Gaussian mixture components defined by centroids.

    Args:
        z (torch.Tensor): Data points (batch_size x feature_dim).
        expert_centroids (torch.Tensor): Centroids of Gaussian mixture components (num_experts x feature_dim).

    Returns:
        torch.Tensor: Log posterior probabilities for each data point (batch_size x num_experts).
    """
    return (
        torch.matmul(z, expert_centroids.t())
        # - 0.5 * (
        #     torch.einsum('ni,ni->n', z, z)[:, None] +
        #     torch.einsum('ni,ni->n', expert_centroids, expert_centroids)[None, :]
        # )
    )


@torch.jit.script
def compute_gating(
    k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor
):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        probs (torch.Tensor): Probability values for each expert (batch_size x num_experts).
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class top_k_gating(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        top_k,
        acc_aux_loss=False,
        dropout=0.1,
        hidden_size=256,
        sample_topk=0,
        aux_loss="mi",
        gate_type="mlp",
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k
        assert sample_topk <= top_k
        self.sample_topk = sample_topk

        self.acc_aux_loss = acc_aux_loss
        self.aux_loss = aux_loss
        self.init_aux_statistics()

        self.gate_type = gate_type
        if gate_type == "mlp":
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_experts, bias=False),
            )
        elif gate_type == "linear":
            self.w_gate = nn.Sequential(nn.Linear(input_size, num_experts, bias=False))
        elif gate_type == "gmm":
            self.w_gate = nn.Linear(input_size, hidden_size, bias=False)
            self.expert_centroids = nn.Parameter(torch.empty(num_experts, hidden_size))
            nn.init.normal_(self.expert_centroids)
            self.temperature = nn.Parameter(torch.zeros(1))
        else:
            print(gate_type)
            raise NotImplementedError

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return "k={}, num_experts={}, aux_loss={}".format(
            self.top_k, self.num_experts, self.aux_loss
        )

    def init_aux_statistics(self):
        """
        Initialize auxiliary statistics based on the chosen auxiliary loss type.
        """
        if self.aux_loss == "mi":
            self.p_e = 0.0
            self.neg_H_e_given_x = 0.0
            self.count_layers = 0
        else:
            self.acc_probs = 0.0
            self.acc_freq = 0.0
            self.acc_lsesq = 0.0
            self.acc_count = 0

    def update_aux_statistics(self, probs, logits, gates, skip_mask=None):
        """
        Update auxiliary statistics based on the current batch.

        Args:
            probs (torch.Tensor): Probability values for each expert.
            logits (torch.Tensor): Logits values for each expert.
            gates (torch.Tensor): Gating values for each expert.
            skip_mask (torch.Tensor): Skip mask tensor.

        """
        if self.aux_loss == "mi":
            log_prob = torch.log_softmax(logits, dim=-1)
            self.p_e = self.p_e + probs.mean(0)
            self.neg_H_e_given_x = self.neg_H_e_given_x + (
                probs * log_prob
            ).sum() / probs.size(0)
            self.count_layers += 1
        else:
            self.acc_count = self.acc_count + logits.size(0)
            self.acc_probs = self.acc_probs + probs.sum(0)
            self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
            lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
            self.acc_lsesq = self.acc_lsesq + lsesq.sum()

    def get_aux_loss_and_clear(self, eps=1e-8):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        if self.aux_loss == "mi":
            denominator = self.count_layers
            p_e = self.p_e / denominator
            H_e = -(p_e * (p_e + eps).log()).sum()
            neg_H_e_given_x = self.neg_H_e_given_x / denominator
            miloss = -(neg_H_e_given_x + H_e)
            loss = miloss
        else:
            switchloss = (
                self.num_experts
                * (
                    F.normalize(self.acc_probs, p=1, dim=0)
                    * F.normalize(self.acc_freq, p=1, dim=0)
                ).sum()
            )
            zloss = self.acc_lsesq / self.acc_count
            loss = switchloss + 0.1 * zloss

        self.init_aux_statistics()
        return loss

    def forward(self, x, skip_mask=None):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        if self.gate_type in ["linear", "mlp"]:
            logits = self.w_gate(x)
        elif self.gate_type == "gmm":
            z = self.w_gate(x)
            logits = (
                log_gmm_posterior(
                    F.normalize(z, p=2, dim=-1),
                    F.normalize(self.expert_centroids, p=2, dim=-1),
                )
                * self.temperature.exp()
            )

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, (skip_mask == 0), 0)
            logits = torch.masked_fill(logits, (skip_mask == 0), 0)

        if self.training and (self.sample_topk > 0):
            _, top_km1_indices = probs.topk(self.top_k - self.sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, self.sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.top_k, dim=1)

        # if self.top_k > 1:
        #     top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

        # gate = torch.zeros_like(top_k_gates)
        # gate[:, 0] = 1
        # top_k_gates = (gate - top_k_gates).detach() + top_k_gates

        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.update_aux_statistics(probs, logits, gates, skip_mask)
        if not self.acc_aux_loss:
            self.loss = self.get_aux_loss_and_clear()
        else:
            self.loss = 0

        return top_k_indices, top_k_gates, probs

        # batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
        #     compute_gating(self.top_k, probs, top_k_gates, top_k_indices)

        # return batch_gates, batch_index, expert_size.tolist(), gates, index_sorted_experts


class ParallelLinear(torch.autograd.Function):
    """
    A custom autograd function for Parallel Linear operation.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size_list, weight, bias=None):
        """
        Forward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        # expert_size_list: List[int] = expert_size.tolist()
        output = ParallelLinear.forward_scriptable(
            input, expert_size_list, weight, bias
        )
        # assert torch.allclose(ParallelLinear._forward_scriptable(input, expert_size, weight, bias),  output)
        ctx.save_for_backward(input, weight, bias)
        ctx.expert_size_list = expert_size_list
        return output

    @staticmethod
    @torch.jit.script
    def forward_scriptable(
        input: Tensor,
        expert_size_list: List[int],
        weight: Tensor,
        bias: Optional[Tensor],
    ):
        """
        Scriptable forward pass of the ParallelLinear operation.

        Args:
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tensor: Output tensor.
        """
        output_buf: Tensor = torch.empty(
            (input.size(0), weight.size(2)), device=input.device, dtype=input.dtype
        )
        num_linears = weight.size(0)

        input_list = input.split(expert_size_list, dim=0)
        output_buf_list = output_buf.split(expert_size_list)

        for i in range(num_linears):
            torch.mm(input_list[i], weight[i], out=output_buf_list[i])

        if bias is not None:
            for i in range(num_linears):
                output_buf_list[i].add_(bias[i])

        output = output_buf
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        """
        Backward pass of the ParallelLinear operation.

        Args:
            ctx: Context object.
            grad_out (Tensor): Gradient of the output.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        input, weight, bias = ctx.saved_tensors
        expert_size_list = ctx.expert_size_list
        return ParallelLinear.backward_scriptable(
            grad_out, input, expert_size_list, weight, bias
        )

    @staticmethod
    @torch.jit.script
    def backward_scriptable(
        grad_out: Tensor,
        input: Tensor,
        expert_size_list: List[int],
        weight: Tensor,
        bias: Optional[Tensor],
    ):
        """
        Scriptable backward pass of the ParallelLinear operation.

        Args:
            grad_out (Tensor): Gradient of the output.
            input (Tensor): Input tensor.
            expert_size_list (List[int]): List of expert sizes.
            weight (Tensor): Weight tensor.
            bias (Optional[Tensor]): Bias tensor.

        Returns:
            Tuple of Tensors: Gradients with respect to input, weight, and bias.
        """
        num_linears = weight.size(0)
        input_list = input.t().split(expert_size_list, dim=1)
        grad_list = grad_out.split(expert_size_list, dim=0)

        d_input_buf = torch.empty_like(input)
        d_input_buf_list = d_input_buf.split(expert_size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        weight_t = weight.permute(0, 2, 1)

        for i in range(num_linears):
            torch.mm(grad_list[i], weight_t[i], out=d_input_buf_list[i])
            torch.mm(input_list[i], grad_list[i], out=d_weight_buf[i])

        d_input = d_input_buf
        d_weight = d_weight_buf

        if bias is not None:
            d_bias_buf = torch.empty_like(bias)
            for i in range(num_linears):
                torch.sum(grad_list[i], dim=0, keepdim=False, out=d_bias_buf[i])
            d_bias = d_bias_buf
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        """
        Initialize the ParallelExperts module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        # self.input_experts = nn.ModuleList(
        #     [nn.Linear(input_size, output_size, bias=bias) for _ in range(num_experts)]
        # )
        self.weight = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.bias = None
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        # std = math.sqrt(2.0 / float(self.weight.size(1) + self.weight.size(2)))
        # a = math.sqrt(3.0) * std
        nn.init.uniform_(
            self.weight, -1.0 / self.weight.size(1), 1.0 / self.weight.size(1)
        )
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        results = ParallelLinear.apply(inputs, expert_size, self.weight, self.bias)
        # expert_size_list: List[int] = expert_size.tolist()
        # input_list = inputs.split(expert_size_list, dim=0)
        # output_list = []
        # for i in range(self.num_experts):
        #     output_list.append(self.input_experts[i](input_list[i]))
        # results = torch.cat(output_list, dim=0)
        return results


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.


    Args:
        input_size: integer - size of the input
        head_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        acc_aux_loss: a boolean - whether to accumulate auxiliary loss
        hidden_size: an integer - hidden size of the experts
        gating_dropout: a float - dropout rate for gating network
        sample_topk: an integer - how many experts to sample during training
        gating_size: an integer - size of the gating network
        aux_loss: a string - type of auxiliary loss ('mi' or 'sparse')
        gate_type: a string - type of gating mechanism ('mlp' or 'topk')
    """

    def __init__(
        self,
        input_size,
        head_size,
        num_experts,
        top_k,
        bias=False,
        activation=None,
        acc_aux_loss=False,
        hidden_size=None,
        gating_dropout=0.0,
        sample_topk=0,
        gating_size=256,
        aux_loss="mi",
        gate_type="mlp",
    ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        if hidden_size is None:
            hidden_size = head_size
        self.output_experts = ParallelExperts(
            num_experts, hidden_size, input_size, bias
        )
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.gate = top_k_gating(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
            acc_aux_loss=acc_aux_loss,
            dropout=gating_dropout,
            sample_topk=sample_topk,
            hidden_size=gating_size,
            aux_loss=aux_loss,
            gate_type=gate_type,
        )

    def extra_repr(self):
        return "k={}".format(self.top_k)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, moe_inp, skip_mask=None):
        """
        Compute gating for the mixture of experts.

        Args:
            moe_inp (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.

        Returns:
            float: Gating loss.
        """

        top_k_indices, top_k_gates, probs = self.gate(moe_inp, skip_mask=skip_mask)
        (
            self.batch_gates,
            self.batch_index,
            expert_size,
            self.index_sorted_experts,
        ) = compute_gating(self.top_k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size.tolist()
        return self.gate.loss, expert_size

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor. (token_num, emb_size)
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        loss, load = self.compute_gate(x)
        # batch_index: for all the tokens (including the copied ones), which ids are they in?
        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (x.shape[0], self.input_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        # y = y.view(-1, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss, load

    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        """

        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        if skip_mask is not None:
            assert (
                x.size()[:-1] == skip_mask.size()
            ), "Skip mask should be same shape as `x`"
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss, _ = self.compute_gate(x, skip_mask)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros(
            (bsz * length * self.top_k, self.head_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """

        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros(
            (bsz * length, self.input_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device,
        )
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y


@torch.jit.script
def stickbreaking_att(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    cum_weight: torch.Tensor,
    att_mask: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute stick-breaking attention weights.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        mask (torch.Tensor): Mask tensor.
        cum_weight (torch.Tensor): Cumulative weight tensor.
        att_mask (Optional[torch.FloatTensor]): Attention mask tensor (default: None).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and attention weights.
    """
    logits = torch.einsum("bikhd,bjhd->bkhij", q, k) / math.sqrt(k.size(-1))
    mask = (mask[None, None, None, :, :] == 0).expand_as(logits)
    logits = logits + att_mask if att_mask is not None else logits
    z = F.sigmoid(logits).masked_fill(mask, 0)
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0)
    re_cum_log_beta = torch.einsum("bnhij,jk->bnhik", log_beta, cum_weight)
    att = z * re_cum_log_beta.exp()
    y = torch.einsum("bkhij,bjhd->bikhd", att, v)
    return y, att


class ModuleFormerAttention(nn.Module):
    def __init__(self, config):
        """
        Initialize the ModuleFormerAttention module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()

        self.q_proj = MoE(
            input_size=config.n_embd,
            head_size=config.att_hidden,
            num_experts=config.n_att_experts,
            top_k=config.k_att,
            acc_aux_loss=False,
            bias=False,
            gating_dropout=config.moe_pdrop,
            sample_topk=config.sample_topk,
            gating_size=config.gating_size,
            aux_loss=config.aux_loss_type,
            gate_type=config.gate_type,
        )
        if config.att_hidden == config.n_embd and config.n_head == 1:
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
        else:
            self.k_proj = nn.Linear(config.n_embd, config.att_hidden)
            self.v_proj = nn.Linear(config.n_embd, config.att_hidden)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.context_length = config.history_length + config.block_size

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(self.context_length, self.context_length, dtype=torch.int8)
            ),
        )
        self.register_buffer(
            "cum_weight",
            torch.tril(torch.ones(self.context_length, self.context_length), -1),
        )
        self.n_head = config.n_head
        self.top_k = config.k_att
        self.n_embd = config.n_embd
        self.att_hidden = config.att_hidden
        self.head_size = config.att_hidden // config.n_head

    def add_history(self, k, v, hidden, use_cache=False):
        """
        Add history to key and value tensors.

        Args:
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            hidden: Hidden state.
            use_cache (bool): Whether to use cached history.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Updated key, value, and history.
        """
        if hidden is None or not use_cache:
            new_k = k
            new_v = v
        else:
            k_history, v_history = hidden
            new_k = torch.cat([k_history, k], dim=1)
            new_v = torch.cat([v_history, v], dim=1)
        k_history = new_k.detach()
        v_history = new_v.detach()

        return new_k, new_v, (k_history, v_history)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        """
        Forward pass of the ModuleFormerAttention module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            head_mask (Optional[torch.FloatTensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[...]]]: Tuple containing outputs.
        """
        (
            B,
            T,
            C,
        ) = (
            hidden_states.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values
        q, aux_loss = self.q_proj.map(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        k, v, hidden = self.add_history(k, v, layer_past, use_cache)
        context_length = k.size(1)

        q = q.view(B, T, self.top_k, self.n_head, self.head_size)  # (B, T, k, nh, hs)
        k = k.view(B, context_length, self.n_head, self.head_size)  # (B, T, nh, hs)
        v = v.view(B, context_length, self.n_head, self.head_size)  # (B, T, nh, hs)

        mask = torch.tril(
            torch.ones(
                context_length, context_length, dtype=torch.int8, device=q.device
            )
        )[context_length - T :, :]
        cum_weight = torch.tril(
            torch.ones(context_length, context_length, device=q.device), -1
        ).type_as(q)

        y, attn_weights = stickbreaking_att(
            q, k, v, mask=mask, cum_weight=cum_weight, att_mask=attention_mask
        )

        # output projection
        y = self.q_proj.reduce(
            y.reshape(B, T, self.top_k, self.att_hidden).type_as(hidden_states)
        )

        y = y.view(B, T, C)  # re-assemble all head outputs side by side

        outputs = (y, hidden, aux_loss)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class ModuleFormerBlock(nn.Module):
    def __init__(self, config):
        """
        Initialize the ModuleFormerBlock module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = ModuleFormerAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = MoE(
            input_size=config.n_embd,
            head_size=config.ffd_hidden,
            num_experts=config.n_mlp_experts,
            top_k=config.k_mlp,
            bias=False,
            activation=get_activation(config.activation_function),
            acc_aux_loss=False,
            gating_dropout=config.moe_pdrop,
            sample_topk=config.sample_topk,
            gating_size=config.gating_size,
            aux_loss=config.aux_loss_type,
            gate_type=config.gate_type,
        )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def get_aux_loss_and_clear(self):
        """
        Get auxiliary loss and clear auxiliary loss accumulators in the attention and MLP layers.

        Returns:
            torch.Tensor: Auxiliary loss.
        """
        return (
            self.attn.q_proj.get_aux_loss_and_clear()
            + self.mlpf.get_aux_loss_and_clear()
        )

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        pad_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        """
        Forward pass of the ModuleFormerBlock module.

        Args:
            hidden_states (Optional[torch.FloatTensor]): Input hidden states.
            layer_past (Optional[Tuple[torch.Tensor]]): Past layer state.
            attention_mask (Optional[torch.FloatTensor]): Attention mask.
            head_mask (Optional[torch.FloatTensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
            Tuple containing outputs or optional attention weights.
        """
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        hidden = attn_outputs[1]
        att_aux_loss = attn_outputs[2]

        hidden_states = hidden_states + self.resid_dropout(attn_output)
        x = self.ln_2(hidden_states)

        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        flat_mask = None
        if self.training and pad_mask is not None:
            # assert (
            #     x.size()[:-1] == skip_mask.size()
            # ), "Skip mask should be same shape as `x`"
            # skip_mask = skip_mask.flatten()[:, None]
            flat_mask = pad_mask.flatten()
            assert x.size(0) == flat_mask.size(0), f"{x.size(0)} != {flat_mask.size(0)}"
            x = x[flat_mask.bool()]

        x_mlp, mlp_aux_loss, gate_load = self.mlpf(x)
        out_size = x_mlp.shape[-1]
        if flat_mask is not None:
            mid = torch.zeros(
                tuple(flat_mask.shape) + (out_size,),
                dtype=x_mlp.dtype,
                device=x_mlp.device,
            )
            mid[flat_mask.bool()] = x_mlp
            x_mlp = mid
        x_mlp = x_mlp.reshape(bsz, length, out_size)
        hidden_states = hidden_states + self.resid_dropout(x_mlp)
        aux_loss = att_aux_loss + mlp_aux_loss
        return (hidden_states, hidden, aux_loss) + attn_outputs[3:] + (gate_load,)


class ModuleFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ModuleFormerConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModuleFormerBlock"]

    def __init__(self, *inputs, **kwargs):
        """
        Initialize the ModuleFormerPreTrainedModel.

        Args:
            *inputs: Variable length input arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*inputs, **kwargs)

        self.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                self._set_gradient_checkpointing(
                    module, True, gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                self._set_gradient_checkpointing(module, False)

    def _set_gradient_checkpointing(
        self,
        module,
        value=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    ):
        """
        Set gradient checkpointing for the ModuleFormerModel.

        Args:
            module: The module for which gradient checkpointing is set.
            value (bool): Whether to enable gradient checkpointing.
        """
        if isinstance(module, ModuleFormerModel):
            module.gradient_checkpointing = value
            module.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


SPARSEGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ModuleFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SPARSEGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoProcenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class MoEBaseModelOutputWithPast(BaseModelOutputWithPast):
    gate_load: Optional[Tuple[list[torch.Tensor]]] = None


@add_start_docstrings(
    "The bare ModuleFormer Model transformer outputting raw hidden-states without any specific head on top.",
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerModel(ModuleFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [ModuleFormerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(
        SPARSEGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Attention mask.
        pad_mask = attention_mask
        if pad_mask is None:
            pad_mask = torch.ones_like(input_ids)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_gate_load = ()
        self.aux_loss = 0
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    pad_mask,
                    **self.gradient_checkpointing_kwargs,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    pad_mask=pad_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            self.aux_loss = self.aux_loss + outputs[2]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[3],)

            # the last element is the gate_load
            all_gate_load += (outputs[-1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            gate_load=all_gate_load,
        )


@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    gate_load: Optional[Tuple[List[torch.Tensor]]] = None


@add_start_docstrings(
    """
    The ModuleFormer Model transformer with a language modeling head on top.
    """,
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerForCausalLM(ModuleFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.causal_mask"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ModuleFormerModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.aux_loss_weight = config.aux_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(
        SPARSEGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            loss = loss.to(hidden_states.dtype)

            if self.aux_loss_weight > 0:
                loss = loss + self.transformer.aux_loss * self.aux_loss_weight

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            gate_load=transformer_outputs.gate_load,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )


@add_start_docstrings(
    """
    The ModuleFormer Model with a sequence classification head on top (linear layer).

    [`ModuleFormerForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    SPARSEGPT_START_DOCSTRING,
)
class ModuleFormerForSequenceClassification(ModuleFormerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        # r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight"
    ]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ModuleFormerModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SPARSEGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
