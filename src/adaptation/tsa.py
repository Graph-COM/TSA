import gc
import logging
import pdb
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.model import GCN, GPRGNN, GSN

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter
from .em import EM
from .lame import LAME
from .t3a import T3A
from .tent import TENT


def scale_nbr_edge_weight(
    nbr_scale: torch.Tensor,
    delta_linear: nn.Module,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    deg_type: str = "ctr_deg",
):
    row, col = edge_index
    mask = row != col
    deg = degree(row)

    log_deg = torch.log(deg + 1)
    max_log_deg = torch.log(deg.max() + 1)
    norm_deg = log_deg / max_log_deg

    if deg_type == "ctr_deg":
        # degree of the center/target/destination nodes
        norm_deg = norm_deg[col].unsqueeze(-1).detach()
    elif deg_type == "sym_deg":
        norm_deg_row = norm_deg[row].unsqueeze(-1).detach()
        norm_deg_col = norm_deg[col].unsqueeze(-1).detach()
        # GCN Normalization sqrt(d_i * d_j) in log domain
        sym_norm_deg = 0.5 * (norm_deg_row + norm_deg_col)
        norm_deg = sym_norm_deg.unsqueeze(-1).detach()

    delta_alpha = (F.sigmoid(delta_linear(norm_deg)) - 0.5).squeeze()
    scale_edge_weight = edge_weight.clone()
    scale_edge_weight[mask] = edge_weight[mask] * (nbr_scale + delta_alpha[mask])
    return scale_edge_weight


@ADAPTER_REGISTRY.register()
class TSA(EM):
    """
    Our proposed method based on Laplacian Regularization.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)
        self.adapter_config = adapter_config
        self.source_stats = source_stats
        self.iter_epochs = adapter_config.iter_epochs
        self.base_tta = adapter_config.base_tta
        self.scale_epochs = adapter_config.scale_epochs
        self.pa_ratio = adapter_config.pa_ratio
        self.scale_lr = adapter_config.scale_lr
        self.scale_label = adapter_config.scale_label
        self.scale_thre = adapter_config.scale_thre
        if isinstance(self.model, GSN) or isinstance(self.model, GCN):
            num_layers = len(self.model.conv)
        elif isinstance(self.model, GPRGNN):
            num_layers = self.model.prop1.K

        self.nbr_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_layers)]
        )
        self.delta_linear = nn.ModuleList()
        for _ in range(num_layers):
            linear_layer = nn.Linear(in_features=1, out_features=1)
            torch.nn.init.zeros_(linear_layer.weight)
            torch.nn.init.zeros_(linear_layer.bias)
            self.delta_linear.append(linear_layer)

    def remove_neighbor_scaling_hook(self):
        if hasattr(self, "_nbr_hooks"):
            for handle in self._nbr_hooks:
                handle.remove()
                del handle
            del self._nbr_hooks

        if isinstance(self.model, GPRGNN):
            if hasattr(self.model.prop1, "_scaled_edge_weights"):
                del self.model.prop1._scaled_edge_weights

    def add_neighbor_scaling_hook(self):
        self._nbr_hooks = []
        if isinstance(self.model, GSN):

            def gsn_hook(i):
                def hook(module, input):
                    x, edge_index, edge_weight = input
                    scaled_weight = scale_nbr_edge_weight(
                        self.nbr_scale[i], self.delta_linear[i], edge_index, edge_weight
                    )
                    # Rewrite the input with the scaled edge_weight (by convention or shared context)
                    return (x, edge_index, scaled_weight)

                return hook

            for i, conv_layer in enumerate(self.model.conv):
                hook_handle = conv_layer.register_forward_pre_hook(gsn_hook(i))
                self._nbr_hooks.append(hook_handle)

        elif isinstance(self.model, GCN):

            def gcn_hook(i):
                def hook(module, input):
                    x, edge_index, edge_weight = input
                    scaled_weight = scale_nbr_edge_weight(
                        self.nbr_scale[i], self.delta_linear[i], edge_index, edge_weight
                    )
                    # Rewrite the input with the scaled edge_weight (by convention or shared context)
                    return (x, edge_index, scaled_weight)

                return hook

            for i, conv_layer in enumerate(self.model.conv):
                hook_handle = conv_layer.register_forward_pre_hook(gcn_hook(i))
                self._nbr_hooks.append(hook_handle)

        elif isinstance(self.model, GPRGNN):

            def gpr_hook(module, input):
                x, edge_index, edge_weight = input
                cached_scaled_weights = []
                # Precompute all scaled edge weights (one for each hop)
                for k in range(module.K):
                    scaled_weights = scale_nbr_edge_weight(
                        self.nbr_scale[k],
                        self.delta_linear[k],
                        edge_index,
                        edge_weight,
                        deg_type="sym_deg",
                    )
                    cached_scaled_weights.append(scaled_weights)

                # Store them in the module for use inside .forward()
                module._scaled_edge_weights = cached_scaled_weights
                # Store them in the module for use inside .forward()
                return input

            hook_handle = self.model.prop1.register_forward_pre_hook(gpr_hook)
            self._nbr_hooks.append(hook_handle)

    def initialize_base_tta(self, help_model):
        if self.base_tta == "TENT":
            return TENT(help_model, self.source_stats, self.adapter_config)
        elif self.base_tta == "LAME":
            return LAME(help_model, self.source_stats, self.adapter_config)
        elif self.base_tta == "T3A":
            return T3A(help_model, self.source_stats, self.adapter_config)
        else:
            raise ValueError(f"Unknown base_tta method: {self.base_tta}")

    def get_TTA(self):
        if self.base_tta == "ERM":
            self.model.eval()
            base_TTA = self.model
        else:
            model_help = deepcopy(self.model)
            base_TTA = self.initialize_base_tta(model_help)

        return base_TTA

    def get_label_and_mask(self, prob):
        label_pred = torch.argmax(prob, -1)
        uncertainty = entropy(prob)
        threshold = self.scale_thre * torch.log(
            torch.tensor(prob.size(1), dtype=torch.float, device=prob.device)
        )
        mask = uncertainty < threshold
        if self.adapter_config.data_name == "MAG":
            # Train without dummy class
            data_mask = label_pred != (prob.size(1) - 1)
        else:
            data_mask = torch.ones_like(label_pred).bool()

        if self.scale_label == "pseudo_label":
            label_one_hot = torch.nn.functional.one_hot(
                label_pred, num_classes=prob.size(1)
            )
            prob = label_one_hot.float()
        return prob.detach(), torch.logical_and(mask, data_mask).detach()

    def adapt(self, data: Data) -> torch.Tensor:
        self.model.to(self.device)
        data = data.to(self.device)
        self.to(self.device)
        src_label_distr = self.source_stats["src_label_distr"].to(self.device)
        src_edge_distr = self.source_stats["src_edge_distr"].to(self.device)
        ###################
        optimizer = optim.Adam(
            list(self.nbr_scale) + list(self.delta_linear.parameters()),
            lr=self.scale_lr,
        )
        ###################
        for epoch in range(self.iter_epochs + 1):
            self.add_neighbor_scaling_hook()
            ###################
            if self.base_tta == "ERM":
                base_TTA = self.get_TTA()
                _, output = base_TTA(data)
                probs = F.softmax(output, dim=-1)
            else:
                base_TTA = self.get_TTA()
                probs = base_TTA.adapt(data)
            ###################

            with torch.no_grad():
                uncertain_indices = self.uncertainty_ranking(probs)
                ctr_label, nbr_label = self.cal_ctr_nbr_label(data, probs)
                tgt_label_distr, tgt_edge_distr = self.cal_tgt_distr(data, probs)

                # E step
                _ = self.graph_align_e_step(
                    data,
                    src_label_distr,
                    tgt_label_distr,
                    src_edge_distr,
                    tgt_edge_distr,
                    ctr_label,
                    nbr_label,
                    uncertain_indices,
                )

            start_time = time.time()
            label, mask = self.get_label_and_mask(probs)
            # logging.info(f"Optimized {mask.sum()/probs.size(0) *100:.2f}% Nodes")
            for scale_epoch in range(self.scale_epochs):
                optimizer.zero_grad()
                _, output = self.model(data)
                erm_prob = F.softmax(output, dim=-1)
                loss = softmax_entropy(label[mask], erm_prob[mask]).mean(0)
                loss.backward()
                optimizer.step()

            # Remove hooks
            self.remove_neighbor_scaling_hook()

        return probs


def softmax_entropy(label_prob: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    epsilon = 1e-5
    return -(label_prob * torch.log(prob + epsilon)).sum(1)


def entropy(input_):
    return -torch.sum(input_ * torch.log(input_ + 1e-9), dim=1)
