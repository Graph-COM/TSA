import logging
import os
import pdb
import random
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree


def set_seed(seed: int) -> None:
    """
    Set the seed for controlling randomness.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Global seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_config_seed(seed, *configs):
    for config in configs:
        config.seed = seed


def replace_nan_with_uniform(tensor: Tensor) -> Tensor:
    # Find rows with NaN values
    nan_mask = torch.isnan(tensor).any(dim=1)
    num_columns = tensor.size(1)
    # Create a uniform distribution row
    uniform_row = torch.full((1, num_columns), 1.0 / num_columns, device=tensor.device)
    tensor[nan_mask] = uniform_row

    return tensor


def edgeidx_to_adj(edge_source, edge_target, num_node):
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)), shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])
    return adj


def num_per_class_split(idx, graph, num_per_class=3):
    """
    Split indices into validation and test set by number of samples per class.
    """
    val_mask = []
    labels = graph.y.numpy()
    num_classes = graph.num_classes
    for cls in range(num_classes):
        cls_indices = idx[labels[idx] == cls]
        if len(cls_indices) < 2 * num_per_class:
            print(
                f"Class {cls} only have {len(cls_indices)} instances to sample {2*num_per_class} instances."
            )
            continue
        sampled_indices = np.random.choice(cls_indices, num_per_class, replace=False)
        val_mask.extend(sampled_indices)

    test_mask = np.setdiff1d(idx, val_mask)
    assert len(set(val_mask)) == len(val_mask)
    return val_mask, test_mask


def to_boolean_mask(idx_mask, total_nodes):
    boolean_mask = np.zeros(total_nodes, dtype=bool)
    boolean_mask[idx_mask] = True
    return torch.from_numpy(boolean_mask)


def nbr_label_prob_0(data: Data, nbr_label: Tensor) -> Tensor:
    """
    Generate a mask that indicates whether a node does not have a neighbor with a specific label.
    Returns a boolean tensor of shape (num_classes, num_nodes) where each row corresponds to a label.
    """
    nbr_label_count = torch.zeros(
        (data.num_classes, data.num_nodes), device=nbr_label.device
    )

    nbr_mask = (
        nbr_label
        == torch.arange(data.num_classes, device=nbr_label.device).unsqueeze(1)
    ).float()
    nbr_label_count.scatter_add_(
        1, data.edge_index[1].unsqueeze(0).expand(data.num_classes, -1), nbr_mask
    )
    nbr_label_mask = nbr_label_count == 0
    return nbr_label_mask


def nbr_label_prob_1(data: Data, nbr_label: Tensor) -> Tensor:
    """
    Generate a mask that indicates whether a node only have neighbors with a specific label.
    Returns a boolean tensor of shape (num_classes, num_nodes) where each row corresponds to a label.
    """
    nbr_label_count = torch.zeros(
        (data.num_classes, data.num_nodes), device=nbr_label.device
    )

    nbr_mask = (
        nbr_label
        == torch.arange(data.num_classes, device=nbr_label.device).unsqueeze(1)
    ).float()
    nbr_label_count.scatter_add_(
        1, data.edge_index[1].unsqueeze(0).expand(data.num_classes, -1), nbr_mask
    )
    degrees = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
    nbr_label_mask = nbr_label_count == degrees
    return nbr_label_mask

class SaveEmb:
    """
    Class object to save model forward feature ebmeddings.
    from "https://github.com/jmiemirza/ActMAD/"
    """
    def __init__(self):
        self.outputs = []
        self.int_mean = []
        self.int_var = []

    def __call__(self, module, module_in, module_out):
        # hook function
        if isinstance(module_out, (tuple, list)):
            self.outputs = module_out[0]
        else:
            self.outputs = module_out

    def clear(self):
        self.outputs = None

    def statistics_update(self):
        self.int_mean.append(torch.mean(self.outputs, dim=0))
        self.int_var.append(torch.var(self.outputs, dim=0))

    def pop_mean(self):
        return torch.mean(torch.stack(self.int_mean), dim=0)

    def pop_var(self):
        return torch.mean(torch.stack(self.int_var), dim=0)