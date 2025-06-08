import logging
import pdb
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree

from src.model import GCN, GPRGNN, GSN

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter
from .lame import LAME
from .t3a import T3A
from .tent import TENT


def snr_adjustment(
    nbr_scale: Tensor,
    delta_linear: nn.Module,
    edge_index: Tensor,
    edge_weight: Tensor,
    deg_type: str = "ctr_deg",
) -> Tensor:
    """
    SNR adjustment to reweight the combination of self-node representations
    and neighborhood-aggregated representations.

    nbr_scale: Scaling bias for k-th layer.
    delta_linear: Linear layer that takes log-normalized degree as input.
    """
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
        # Normalization sqrt(d_i * d_j) in log domain
        sym_norm_deg = 0.5 * (norm_deg_row + norm_deg_col)
        norm_deg = sym_norm_deg.unsqueeze(-1).detach()

    delta_alpha = (F.sigmoid(delta_linear(norm_deg)) - 0.5).squeeze()
    scale_edge_weight = edge_weight.clone()
    scale_edge_weight[mask] = edge_weight[mask] * (nbr_scale + delta_alpha[mask])
    return scale_edge_weight


@ADAPTER_REGISTRY.register()
class TSA(BaseAdapter):
    """
    Test-Time Structural Alignment (TSA).
    TSA variants include TSA-T3A, TSA-TENT, and TSA-LAME.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        super().__init__(pre_model, source_stats, adapter_config)
        self.adapter_config = adapter_config

        # Information for neighborhood alignment
        self.source_stats = source_stats
        self.cal_type = adapter_config.cal_type
        self.pa_ratio = adapter_config.pa_ratio
        # Information for SNR adjustment
        self.scale_lr = adapter_config.scale_lr
        self.scale_thre = adapter_config.scale_thre
        self.scale_epochs = adapter_config.scale_epochs

        self.base_tta = adapter_config.base_tta
        self.iter_epochs = adapter_config.iter_epochs

        if isinstance(self.model, GSN) or isinstance(self.model, GCN):
            num_layers = len(self.model.conv)
        elif isinstance(self.model, GPRGNN):
            num_layers = self.model.prop1.K

        # Parameters to train from SNR adjustment
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
                    scaled_weight = snr_adjustment(
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
                    scaled_weight = snr_adjustment(
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
                    scaled_weights = snr_adjustment(
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

    def cal_ctr_nbr_label(self, data: Data, probs: Tensor):
        if self.cal_type == "predict_target":
            _, pred = torch.max(probs, dim=-1)  # [N]
            ctr_label = pred[data.edge_index[1]]  # [E]
            nbr_label = pred[data.edge_index[0]]  # [E]
        elif self.cal_type in ["true_target"]:
            ctr_label = data.y[data.edge_index[1]]
            nbr_label = data.y[data.edge_index[0]]
        return ctr_label, nbr_label

    def cal_tgt_distr(self, data: Data, probs: Tensor):
        if self.cal_type == "predict_target":
            tgt_edge_distr = self.model.cal_edge_distr_soft(data, probs)
        elif self.cal_type == "true_target":
            tgt_edge_distr = self.model.cal_edge_distr_true(data)
        return tgt_edge_distr

    def uncertainty_ranking(self, prob):
        """
        Return indices that have high uncerrainty and do not do Neighborhood Alignment.
        """
        uncertainty = entropy(prob)
        threshold = self.pa_ratio * torch.log(
            torch.tensor(prob.size(1), dtype=torch.float, device=prob.device)
        )
        mask = uncertainty > threshold
        return mask

    def neighborhood_align(
        self,
        data: Data,
        src_edge_distr: Tensor,
        tgt_edge_distr: Tensor,
        ctr_label: Tensor,
        nbr_label: Tensor,
        uncertain_indices: Tensor,
    ):
        """
        Neighborhood alignment recalibrates the influence of neighboring nodes
        during message aggregation, thereby aligning the target neighborhood
        distribution with the source domain.

        Example of a ratio matrix for a 3-class case:
           [[r_ctr0_nbr0, r_ctr0_nbr1, r_ctr0_nbr2],
           [r_ctr1_nbr0, r_ctr1_nbr1, r_ctr1_nbr2],
           [r_ctr2_nbr0, r_ctr2_nbr1, r_ctr2_nbr2]])
        """
        # --- Align target graph to source graph ---

        ratio = (src_edge_distr) / (tgt_edge_distr)

        degree_n = degree(data.edge_index[1], num_nodes=data.x.size(0))
        degree_ctr = degree_n[data.edge_index[1]]
        edge_weight = ratio[ctr_label, nbr_label]

        ### Do Neighborhood Alignment only on low uncertainty nodes
        if uncertain_indices is not None:
            ctr_uncertain = uncertain_indices[data.edge_index[1]]  # [E]
            nbr_uncertain = uncertain_indices[data.edge_index[0]]  # [E]
            combined_uncertain_mask = ctr_uncertain | nbr_uncertain
            edge_weight[combined_uncertain_mask] = 1.0
            pa_percent = (~combined_uncertain_mask).sum() / data.edge_index.size(1)
            logging.info(f"Neighborhood Alignment {pa_percent*100:.2f}% Edges")

        # if degree=1 than do not change edge weight
        edge_weight[degree_ctr == 1] = 1.0

        w_degree_n = torch.zeros_like(degree_n)
        w_degree_n.scatter_add_(0, data.edge_index[1], edge_weight)
        w_degree_ctr = w_degree_n[data.edge_index[1]]

        weight_deg_edge_weight = edge_weight * degree_ctr / w_degree_ctr
        if torch.any(torch.isnan(weight_deg_edge_weight)):
            weight_deg_edge_weight = safe_mean(
                weight_deg_edge_weight, w_degree_ctr, data, 0.0
            )
        data.edge_weight = weight_deg_edge_weight

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

        label_one_hot = torch.nn.functional.one_hot(
            label_pred, num_classes=prob.size(1)
        )
        prob = label_one_hot.float()
        return prob.detach(), torch.logical_and(mask, data_mask).detach()

    def adapt(self, data: Data) -> Tensor:
        self.model.to(self.device)
        data = data.to(self.device)
        self.to(self.device)

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
                tgt_edge_distr = self.cal_tgt_distr(data, probs)

                # E step
                self.neighborhood_align(
                    data,
                    src_edge_distr,
                    tgt_edge_distr,
                    ctr_label,
                    nbr_label,
                    uncertain_indices,
                )

            label, mask = self.get_label_and_mask(probs)
            # logging.info(f"Optimized {mask.sum()/probs.size(0) *100:.2f}% Nodes")
            for _ in range(self.scale_epochs):
                optimizer.zero_grad()
                _, output = self.model(data)
                erm_prob = F.softmax(output, dim=-1)
                loss = softmax_entropy(label[mask], erm_prob[mask]).mean(0)
                loss.backward()
                optimizer.step()

            # Remove hooks
            self.remove_neighbor_scaling_hook()

        return probs


def safe_mean(edge_weight, w_degree_ctr, data, safe_value=0.0):
    # Mask out NaN and Inf values and set to 0
    mask = ~torch.isnan(edge_weight) & ~torch.isinf(edge_weight)
    print("Number of problematic edges:", (~mask).sum(), edge_weight[~mask])
    valid_tensor = torch.where(
        mask, edge_weight, torch.tensor(safe_value, dtype=edge_weight.dtype)
    )
    # Print logging
    ctr_label = data.y[data.edge_index[1, :][w_degree_ctr == 0]]
    nbr_label = data.y[data.edge_index[0, :][w_degree_ctr == 0]]
    label_pair = torch.unique(torch.stack((ctr_label, nbr_label), dim=1), dim=0)
    pairs_list = [f"{pair[0].item()}-{pair[1].item()}" for pair in label_pair]
    pairs_str = " and ".join(pairs_list)
    logging.info(
        f"The following class pairs (ctr-nbr) are not connected in source graph: {pairs_str}"
    )
    return valid_tensor


def softmax_entropy(label_prob: Tensor, prob: Tensor) -> Tensor:
    """Entropy of softmax distribution from logits."""
    epsilon = 1e-5
    return -(label_prob * torch.log(prob + epsilon)).sum(1)


def entropy(input_):
    return -torch.sum(input_ * torch.log(input_ + 1e-9), dim=1)
