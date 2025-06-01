import logging
import pdb
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import add_remaining_self_loops, degree

from src.utils.utils import replace_nan_with_uniform

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class EM(BaseAdapter):
    """
    Our proposed method based on Laplacian Regularization.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)
        self.init_type = adapter_config.init_type
        self.cal_type = adapter_config.cal_type
        self.reweight = adapter_config.reweight
        self.alpha = adapter_config.alpha
        self.weight_deg_norm = adapter_config.weight_deg_norm
        self.pa_ratio = adapter_config.pa_ratio

    def cal_ctr_nbr_label(self, data: Data, probs: Tensor):
        if self.cal_type == "predict_target":
            _, pred = torch.max(probs, dim=-1)  # [N]
            ctr_label = pred[data.edge_index[1]]  # [E]
            nbr_label = pred[data.edge_index[0]]  # [E]
            # print(torch.bincount(pred))
        elif self.cal_type in ["true_target"]:
            ctr_label = data.y[data.edge_index[1]]
            nbr_label = data.y[data.edge_index[0]]
        return ctr_label, nbr_label

    def cal_tgt_distr(self, data: Data, probs: Tensor):
        if self.cal_type == "predict_target":
            tgt_label_distr = self.model.cal_label_distr_soft(data, probs)
            tgt_edge_distr = self.model.cal_edge_distr_soft(data, probs)
        elif self.cal_type == "true_target":
            tgt_label_distr = self.model.cal_label_distr_true(data)
            tgt_edge_distr = self.model.cal_edge_distr_true(data)
        return tgt_label_distr, tgt_edge_distr

    def init_tgt_distr(self, data: Data):
        if self.init_type.split("_")[-1] == "source":
            self.logger.info("Initilize with source distribution")
            # init_type is "predict_source" or "true_source"
            tgt_label_distr = self.source_stats["src_label_distr"].clone()
            tgt_edge_distr = self.source_stats["src_edge_distr"].clone()
        elif self.init_type == "predict_target":
            self.logger.info("Initilize with predicted target distribution")
            tgt_label_distr = self.model.cal_label_distr_soft(data)
            tgt_edge_distr = self.model.cal_edge_distr_soft(data)
        elif self.init_type == "true_target":
            self.logger.info("Initilize with true target distribution")
            tgt_label_distr = self.model.cal_label_distr_true(data)
            tgt_edge_distr = self.model.cal_edge_distr_true(data)
        return tgt_label_distr, tgt_edge_distr

    def uncertainty_ranking(self, prob):
        """
        Return indices that have high uncerrainty and do not do PairAlign
        """
        uncertainty = entropy(prob)
        # ranking_indices = torch.argsort(uncertainty, descending=True)
        # num_nodes = ranking_indices.size(0)
        # percentile = min(1.0, self.percentile)
        # num_select = int((1 - percentile) * num_nodes)
        # mask = torch.zeros(num_nodes, dtype=torch.bool, device=prob.device)
        # mask[ranking_indices[:num_select]] = True
        # ratio small, pa small, ratio large pa large
        threshold = self.pa_ratio * torch.log(
            torch.tensor(prob.size(1), dtype=torch.float, device=prob.device)
        )
        mask = uncertainty > threshold
        return mask

    def graph_align_e_step(
        self,
        data: Data,
        src_label_distr: Tensor,
        tgt_label_distr: Tensor,
        src_edge_distr: Tensor,
        tgt_edge_distr: Tensor,
        ctr_label: Tensor,
        nbr_label: Tensor,
        uncertain_indices: Tensor,
    ):
        """
        Graph Align + E step of EM algorithm

        Example of a ratio matrix for a 3-class case:
           [[r_ctr0_nbr0, r_ctr0_nbr1, r_ctr0_nbr2],
           [r_ctr1_nbr0, r_ctr1_nbr1, r_ctr1_nbr2],
           [r_ctr2_nbr0, r_ctr2_nbr1, r_ctr2_nbr2]])
        """
        # --- Align target graph to source graph ---

        unnorm_ratio = (src_edge_distr) / (tgt_edge_distr)

        diag_indices = torch.arange(unnorm_ratio.size(0))

        ratio = unnorm_ratio

        degree_n = degree(data.edge_index[1], num_nodes=data.x.size(0))
        degree_ctr = degree_n[data.edge_index[1]]
        ratio = ratio.pow(self.alpha)
        edge_weight = ratio[ctr_label, nbr_label]

        ### Do PairAlign only on low uncertainty nodes
        if uncertain_indices is not None:
            ctr_uncertain = uncertain_indices[data.edge_index[1]]  # [E]
            nbr_uncertain = uncertain_indices[data.edge_index[0]]  # [E]
            combined_uncertain_mask = ctr_uncertain | nbr_uncertain
            edge_weight[combined_uncertain_mask] = 1.0
            pa_percent = (~combined_uncertain_mask).sum() / data.edge_index.size(1)
            logging.info(f"PairAlign {pa_percent*100:.2f}% Edges")

        # if degree=1 than do not change edge weight
        edge_weight[degree_ctr == 1] = 1.0

        w_degree_n = torch.zeros_like(degree_n)
        w_degree_n.scatter_add_(0, data.edge_index[1], edge_weight)
        w_degree_ctr = w_degree_n[data.edge_index[1]]
        # pdb.set_trace()
        if self.weight_deg_norm == "ctr_deg":
            weight_deg_edge_weight = edge_weight * degree_ctr / w_degree_ctr
            if torch.any(torch.isnan(weight_deg_edge_weight)):
                weight_deg_edge_weight = safe_mean(
                    weight_deg_edge_weight, w_degree_ctr, data, 0.0
                )
            data.edge_weight = weight_deg_edge_weight
        else:
            raise ValueError("Support only 'ctr_deg'.")
        # data.edge_weight = edge_weight

        return None


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


def entropy(input_):
    return -torch.sum(input_ * torch.log(input_ + 1e-9), dim=1)
