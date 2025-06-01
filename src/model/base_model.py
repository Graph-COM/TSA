import pdb
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops, degree, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

from src.utils import (
    Metrics,
    SaveEmb,
    nbr_label_prob_0,
    nbr_label_prob_1,
    replace_nan_with_uniform,
)


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    """
    Computes the symmetric normalization of the adjacency matrix for GCNs.
    Different from pyg, edge weights are normalized using the degrees computed from the current edge list.
    """
    fill_value = 2.0 if improved else 1.0

    assert flow in ["source_to_target", "target_to_source"]
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = degree(row, num_nodes).float()
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class BaseModel(nn.Module):
    def __init__(self, metrics: Metrics, model_config: DictConfig):
        super().__init__()
        # Initialize your model's parameters here
        self.model_config = model_config
        self.epochs = model_config.epochs
        self.learning_rate = model_config.learning_rate
        self.opt_decay_step = model_config.opt_decay_step
        self.opt_decay_rate = model_config.opt_decay_rate
        self.device = torch.device(
            f"{self.model_config.device}" if torch.cuda.is_available() else "cpu"
        )
        bn_name = "_bnf" if self.model_config.bn_feature else ""
        bn_name = "_bnc" + bn_name if self.model_config.bn_classifier else bn_name
        self.model_path = (
            Path(self.model_config.root)
            / self.model_config.data_name
            / f"model_{self.model_config.name}_{self.model_config.source}_{self.model_config.train_type}{bn_name}_{self.model_config.seed}.pt"
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        self.iscalibrated = False

    def forward(self, data):
        # Define the forward pass
        raise NotImplementedError

    def fit(self, data: Data):
        self.to(self.device)
        data = data.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.opt_decay_step, gamma=self.opt_decay_rate
        )
        if self.model_config.train_type == "bal":
            class_counts = torch.bincount(data.y[data.src_train_mask])
            class_weights = 1.0 / class_counts.float()
            class_weights[torch.isinf(class_weights)] = 0
            class_weights = class_weights / class_weights.sum()
            print(class_weights)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.model_config.train_type == "no_bal":
            criterion = nn.CrossEntropyLoss()
        # pdb.set_trace()
        # data.src_train_mask = torch.logical_and(data.src_train_mask, data.y != 19)
        # data.src_val_mask = torch.logical_and(data.src_val_mask, data.y != 19)
        # data.src_test_mask = torch.logical_and(data.src_test_mask, data.y != 19)
        for epoch in range(self.epochs):
            self.train()  # Set the model to training mode
            optimizer.zero_grad()
            _, outputs = self(data)
            loss = criterion(outputs[data.src_train_mask], data.y[data.src_train_mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss = loss.item()

            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                _, outputs = self(data)
                val_loss = criterion(
                    outputs[data.src_val_mask],
                    data.y[data.src_val_mask],
                )
                _, predicted = torch.max(outputs, 1)

                val_accuracy = (
                    (predicted[data.src_val_mask] == data.y[data.src_val_mask])
                    .float()
                    .mean()
                    .item()
                )

                labels = data.y[data.src_val_mask].cpu().numpy()
                predicted = predicted[data.src_val_mask].cpu().numpy()
                val_bal_accuracy = balanced_accuracy_score(labels, predicted)
                val_f1 = f1_score(
                    labels,
                    predicted,
                    average="macro" if outputs.size(1) > 2 else "binary",
                )

            print(
                f"Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}, Val Bal Accuracy: {val_bal_accuracy:.5f}, Val F1: {val_f1:.5f}"
            )

        torch.save(self.state_dict(), self.model_path)

    def predict(self, data: Data, mask: Tensor, mask_name: str):
        self.eval()
        self.to(self.device)
        data = data.to(self.device)
        # pdb.set_trace()
        with torch.no_grad():
            _, outputs = self(data)
            probs = F.softmax(outputs, dim=-1)
            self.metrics.calculate_metrics(probs[mask], data.y[mask], mask_name)
        return probs

    def calibrate(self, data: Data):
        self.iscalibrated = True
        self.register_parameter("T", torch.nn.Parameter(torch.ones(1)))
        self.register_parameter("b", torch.nn.Parameter(torch.zeros(data.num_classes)))
        bcts_optimizer = torch.optim.Adam([self.T, self.b], lr=0.1)
        self.eval()
        self.to(self.device)
        data = data.to(self.device)
        class_counts = torch.bincount(data.y[data.src_val_mask])
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        for epoch in range(50):
            bcts_optimizer.zero_grad()
            _, outputs = self(data)
            loss = criterion(outputs[data.src_val_mask], data.y[data.src_val_mask])
            loss.backward()
            bcts_optimizer.step()
            cal_loss = loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)

                val_accuracy = (
                    (predicted[data.src_val_mask] == data.y[data.src_val_mask])
                    .float()
                    .mean()
                    .item()
                )

            # print(
            #     f"Calibration Epoch {epoch+1}, Cal/Val Loss: {cal_loss}, Val Accuracy: {val_accuracy}"
            # )
        print(f"self.T {self.T}, self.b {self.b}")

    def calibrated(self, logits):
        return (logits / self.T) + self.b

    def get_pretrain(self, data: Data):
        """
        Pretrain the model on the source domain.
        """
        if self.model_path.exists():
            self.load_state_dict(
                torch.load(self.model_path, map_location=self.device), strict=False
            )
            self.predict(data, data.src_val_mask, "Val")
            probs = self.predict(data, data.src_test_mask, "Test")
            torch.save(probs, "source_graph.pred.pt")
        else:
            self.fit(data)
            self.predict(data, data.src_val_mask, "Val")
            self.predict(data, data.src_test_mask, "Test")

    def get_src_stats(self, data: Data):
        """
        Compute the source statistics for the source domain.
        """
        if self.model_config.source_stats in ["true_source", "true_target"]:
            src_label_distr = self.cal_label_distr_true(data)
            src_edge_distr = self.cal_edge_distr_true(data)
            true_src_label_distr, true_src_edge_distr = (
                src_label_distr.clone,
                src_edge_distr.clone(),
            )
        else:
            src_label_distr = self.cal_label_distr_soft(data)
            src_edge_distr = self.cal_edge_distr_soft(data)
            true_src_label_distr = self.cal_label_distr_true(data)
            true_src_edge_distr = self.cal_edge_distr_true(data)

        custom_stats = self._custom_src_stats(data) or {}
        return {
            "src_label_distr": src_label_distr,
            "src_edge_distr": src_edge_distr,
            "true_src_label_distr": true_src_label_distr,
            "true_src_edge_distr": true_src_edge_distr,
            **custom_stats,  # merge custom dictionary
        }

    def _custom_src_stats(self, data: Data):
        """
        Optional hook for custom stats, should return a dict.
        """
        return {}

    def cal_label_distr_soft(self, data, probs: Optional[Tensor] = None):
        """
        Compute the label distribution with soft probabilities.
        """
        if probs is None:
            self.eval()
            self.to(self.device)
            data = data.to(self.device)
            _, outputs = self(data)
            probs = F.softmax(outputs, dim=-1)

        return torch.mean(probs, dim=0)

    def cal_label_distr_true(self, data):
        """
        Compute the true label distribution using node label.
        """
        data = data.to(self.device)
        probs = F.one_hot(data.y, num_classes=data.num_classes)
        return torch.mean(probs.float(), dim=0)

    def cal_edge_distr_hard(self, data: Data, probs: Optional[Tensor] = None):
        """
        Compute the edge distribution with hard pseudo labels for center nodes.

            Notes:
                - The target node is the center node in aggregation.
                - The target node indices are given by `data.edge_index[1]`.
        """
        if probs is None:
            self.eval()
            self.to(self.device)
            data = data.to(self.device)
            _, outputs = self(data)
            probs = F.softmax(outputs, dim=-1)

        ctr_probs = probs[data.edge_index[1]]
        _, ctr_label = torch.max(ctr_probs, 1)
        nbr_probs = probs[data.edge_index[0]]

        sum_probs = torch.zeros(
            (data.num_classes, data.num_classes), device=self.device
        )  # Shape [3, 3]
        sum_probs.scatter_add_(
            0, ctr_label.unsqueeze(1).expand(-1, nbr_probs.size(1)), nbr_probs
        )
        counts = torch.bincount(ctr_label, minlength=data.num_classes)
        counts = counts.float()
        edge_probs = sum_probs / counts.unsqueeze(1)
        # edge_probs = replace_nan_with_uniform(edge_probs)
        return edge_probs

    def cal_edge_distr_soft(self, data: Data, probs: Optional[Tensor] = None):
        """
        Compute the edge distribution with soft probabilities for center nodes.

            Notes:
                - The target node is the center node in aggregation.
                - The target node indices are given by `data.edge_index[1]`.
        """
        if probs is None:
            self.eval()
            self.to(self.device)
            data = data.to(self.device)
            _, outputs = self(data)
            probs = F.softmax(outputs, dim=-1)

        ctr_probs = probs[data.edge_index[1]]
        nbr_probs = probs[data.edge_index[0]]

        # Compute the outer products for each pair of vectors and sum them all
        outer_products = ctr_probs.unsqueeze(2) * nbr_probs.unsqueeze(1)
        # outer_products = data.edge_weight.view(-1, 1, 1) * outer_products
        edge_probs = outer_products.sum(dim=0)
        edge_probs_sum = edge_probs.sum(dim=1).unsqueeze(1)
        return edge_probs / edge_probs_sum

    def cal_edge_distr_true(self, data: Data, degree_one: bool = True):
        """
        Compute the true edge distribution using node label

            Notes:
                - The target node is the center node in aggregation.
                - The target node indices are given by `data.edge_index[1]`.
        """
        self.eval()
        self.to(self.device)
        data = data.to(self.device)
        probs = F.one_hot(data.y, num_classes=data.num_classes)
        ctr_probs = probs[data.edge_index[1]]
        _, ctr_label = torch.max(ctr_probs, 1)

        nbr_probs = probs[data.edge_index[0]]
        _, nbr_label = torch.max(nbr_probs, 1)

        if not degree_one:
            degrees = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            # Filter out degree one edges
            not_degree_one = degrees != 1
            mask = not_degree_one[data.edge_index[1]]
        else:
            mask = torch.ones_like(ctr_label).bool()

        sum_probs = torch.zeros(
            (data.num_classes, data.num_classes), device=self.device
        )  # Shape [3, 3]
        sum_probs.scatter_add_(
            0,
            ctr_label[mask].unsqueeze(1).expand(-1, nbr_probs[mask].size(1)),
            nbr_probs[mask].float(),
        )

        # # Do not contain prob0 and prob1
        # nbr_label_mask = ~torch.logical_or(
        #     nbr_label_prob_1(data, nbr_label), nbr_label_prob_0(data, nbr_label)
        # )
        # counts = torch.zeros((data.num_classes, data.num_classes), device=self.device)
        # for i in range(data.num_classes):
        #     for j in range(data.num_classes):
        #         # pdb.set_trace()
        #         ctr_nbr_label_mask = nbr_label_mask[j][data.edge_index[1]]
        #         counts[i, j] = (
        #             torch.logical_and(ctr_nbr_label_mask, ctr_label == i).sum().item()
        #         )

        # # Subtract prob1 that was counted in sum_probs
        # sum_deg_1 = torch.zeros(
        #     (data.num_classes, data.num_classes), device=self.device
        # )
        # for i in range(data.num_classes):
        #     for j in range(data.num_classes):
        #         deg_1_mask = nbr_label_prob_1(data, nbr_label)[j][data.edge_index[1]]
        #         sum_deg_1[i, j] = (
        #             torch.logical_and(deg_1_mask, ctr_label == i).sum().item()
        #         )

        # edge_probs = (sum_probs - sum_deg_1) / counts

        # edge_probs = (sum_probs) / counts
        # pdb.set_trace()
        # expanded_degrees = degrees.unsqueeze(0).expand(nbr_label_count_mask.size(0), -1)
        # masked_degrees = expanded_degrees[nbr_label_count_mask.bool()]

        counts = torch.bincount(ctr_label[mask], minlength=data.num_classes)
        counts = counts.float()
        edge_probs = sum_probs / counts.unsqueeze(1)
        # pdb.set_trace()
        return edge_probs
