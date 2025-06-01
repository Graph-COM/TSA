import logging
import pdb
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from torch_sparse import coalesce

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class GTrans(BaseAdapter):
    """
    Our proposed method based on Laplacian Regularization.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)

        self.lr_feat = adapter_config.lr_feat
        self.lr_adj = adapter_config.lr_adj
        self.epochs = adapter_config.epochs
        self.ratio = adapter_config.ratio
        self.loop_feat = adapter_config.loop_feat
        self.loop_adj = adapter_config.loop_adj
        ##############################
        self.max_final_samples = 5
        self.eps = 1e-7
        self.modified_edge_index: torch.Tensor = None
        self.perturbed_edge_weight: torch.Tensor = None
        self.epochs_resampling = self.epochs
        self.do_synchronize = True
        self.model.requires_grad_(False)

    def adapt(self, data: Data) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)
        data = data.to(self.device)

        self.logger = logging.getLogger(__name__)
        self.nnodes = data.x.size(0)
        self.input_dim = data.x.size(1)
        delta_feat = Parameter(
            torch.FloatTensor(self.nnodes, self.input_dim).to(self.device)
        )
        self.delta_feat = delta_feat
        delta_feat.data.fill_(1e-7)
        self.optimizer_feat = torch.optim.Adam([delta_feat], lr=self.lr_feat)
        self.data = data

        n_perturbations = int(self.ratio * data.edge_index.shape[1] // 2)
        print("n_perturbations:", n_perturbations)
        self.sample_random_block()

        self.perturbed_edge_weight.requires_grad = True
        self.optimizer_adj = torch.optim.Adam(
            [self.perturbed_edge_weight], lr=self.lr_adj
        )

        feat = data.x
        edge_index, edge_weight = data.edge_index, data.edge_weight
        for it in range(self.epochs // (self.loop_feat + self.loop_adj)):
            for loop_feat in range(self.loop_feat):
                self.optimizer_feat.zero_grad()
                loss = self.test_time_loss(feat + delta_feat, edge_index, edge_weight)
                loss.backward()

                if loop_feat == 0:
                    print(f"Epoch {it}, Loop Feat {loop_feat}: {loss.item()}")

                self.optimizer_feat.step()

            new_feat = (feat + delta_feat).detach()
            for loop_adj in range(self.loop_adj):
                self.perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight = self.get_modified_adj()
                if torch.cuda.is_available() and self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                loss = self.test_time_loss(new_feat, edge_index, edge_weight)

                gradient = grad_with_checkpoint(loss, self.perturbed_edge_weight)[0]

                if loop_adj == 0:
                    print(f"Epoch {it}, Loop Adj {loop_adj}: {loss.item()}")

                with torch.no_grad():
                    self.update_edge_weights(n_perturbations, it, gradient)
                    self.perturbed_edge_weight = self.project(
                        n_perturbations, self.perturbed_edge_weight, self.eps
                    )
                    del edge_index, edge_weight

                if it < self.epochs_resampling - 1:
                    self.perturbed_edge_weight.requires_grad = True
                    self.optimizer_adj = torch.optim.Adam(
                        [self.perturbed_edge_weight], lr=self.lr_adj
                    )

            if self.loop_adj != 0:
                edge_index, edge_weight = self.get_modified_adj()
                edge_weight = edge_weight.detach()

        print(f"Epoch {it+1}: {loss}")
        if self.loop_adj != 0:
            edge_index, edge_weight = self.sample_final_edges(n_perturbations, data)

        with torch.no_grad():
            loss = self.test_time_loss(feat + delta_feat, edge_index, edge_weight)
        print("final loss:", loss.item())

        data.x = feat + delta_feat
        data.edge_index = edge_index
        data.edge_weight = edge_weight
        _, output = self.model(data)
        probs = F.softmax(output, dim=-1)
        print("Test:")

        tgt_edge_distr = self.model.cal_edge_distr_soft(data, probs)
        print("final label:\n", probs.mean(dim=0))
        # print("final edge:\n", tgt_edge_distr)

        return probs

    def get_modified_adj(self):
        # A' =  \oplus \Delta_A is the paper
        # self.modified_edge_index and self.perturbed_edge_weight are directed

        modified_edge_index, modified_edge_weight = to_symmetric(
            self.modified_edge_index, self.perturbed_edge_weight, self.nnodes
        )
        edge_index = torch.cat((self.data.edge_index, modified_edge_index), dim=-1)
        edge_weight = torch.cat((self.data.edge_weight, modified_edge_weight))

        edge_index, edge_weight = coalesce(
            edge_index, edge_weight, m=self.nnodes, n=self.nnodes, op="sum"
        )
        # Allow removal of edges
        edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
        return edge_index, edge_weight

    def augument_predict(self, feat, edge_index, edge_weight):
        original_feat, original_edge_index, original_edge_weight = (
            self.data.x,
            self.data.edge_index,
            self.data.edge_weight,
        )
        self.data.x, self.data.edge_index, self.data.edge_weight = (
            feat,
            edge_index,
            edge_weight,
        )
        Z, Y = self.model(self.data)
        self.data.x, self.data.edge_index, self.data.edge_weight = (
            original_feat,
            original_edge_index,
            original_edge_weight,
        )
        return Z, Y

    def test_time_loss(self, feat, edge_index, edge_weight):
        loss = 0
        output1 = self.augment(
            feat=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            strategy="dropedge",
            p=0.05,
        )
        output2 = self.augment(
            feat=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            strategy="dropedge",
            p=0.0,
        )
        output3 = self.augment(
            feat=feat,
            edge_index=edge_index,
            edge_weight=edge_weight,
            strategy="shuffle",
        )

        loss = inner(output1, output2) - inner(output2, output3)
        return loss

    def augment(self, feat, edge_index, edge_weight, strategy="dropedge", p=0.5):
        if strategy == "shuffle":
            idx = torch.randperm(feat.shape[0])
            feat = feat[idx, :]
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)

        if strategy == "dropnode":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)

        if strategy == "dropmix":
            feat = self.data.x + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)

        if strategy == "dropfeat":
            feat = F.dropout(self.feat, p=p) + self.delta_feat

        Z, _ = self.augument_predict(feat, edge_index, edge_weight)
        return Z

    def sample_random_block(self):
        edge_index = self.data.edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        row, col = edge_index[0], edge_index[1]
        edge_index_id = (
            (2 * self.nnodes - row - 1) * row // 2 + col - row - 1
        )  # // is important to get the correct result
        edge_index_id = edge_index_id.long()
        self.current_search_space = edge_index_id
        self.modified_edge_index = linear_to_triu_idx(
            self.nnodes, self.current_search_space
        )
        self.perturbed_edge_weight = torch.full_like(
            self.current_search_space,
            self.eps,
            dtype=torch.float32,
            requires_grad=True,
        )
        return

    @torch.no_grad()
    def sample_final_edges(self, n_perturbations, data):
        best_loss = float("Inf")
        perturbed_edge_weight = self.perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= self.eps] = 0
        feat = data.x
        for i in range(self.max_final_samples):
            if best_loss == float("Inf"):
                # In first iteration employ top k heuristic instead of sampling
                # Sample a ratio of the total edges/2 -> These edges will be removed
                sampled_edges = torch.zeros_like(perturbed_edge_weight)
                sampled_edges[
                    torch.topk(perturbed_edge_weight, n_perturbations).indices
                ] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                print(f"{i}-th sampling: too many samples {n_samples}")
                continue
            self.perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = self.get_modified_adj()
            with torch.no_grad():
                loss = self.test_time_loss(feat, edge_index, edge_weight)
            # Save best sample
            if best_loss > loss:
                best_loss = loss
                print(f"best_loss at {i} final samples:", best_loss)
                best_edges = self.perturbed_edge_weight.clone().cpu()

        # Recover best sample
        self.perturbed_edge_weight.data.copy_(best_edges.to(self.device))
        edge_index, edge_weight = self.get_modified_adj()
        edge_mask = edge_weight == 1

        allowed_perturbations = 2 * n_perturbations

        edges_after_attack = edge_mask.sum()
        clean_edges = self.data.edge_index.shape[1]
        # assert (
        #     edges_after_attack >= clean_edges - allowed_perturbations
        #     and edges_after_attack <= clean_edges + allowed_perturbations
        # ), f"{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations"
        return edge_index[:, edge_mask], edge_weight[edge_mask]

    def project(self, n_perturbations, values, eps, inplace=False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(values - miu, min=eps, max=1 - eps))
        else:
            values.data.copy_(torch.clamp(values, min=eps, max=1 - eps))
        return values

    def update_edge_weights(self, n_perturbations, epoch, gradient):
        self.optimizer_adj.zero_grad()
        self.perturbed_edge_weight.grad = gradient
        self.optimizer_adj.step()
        self.perturbed_edge_weight.data[self.perturbed_edge_weight < self.eps] = (
            self.eps
        )


def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (
        n
        - 2
        - torch.floor(
            torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5
        )
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1
        - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))


def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if func(miu) == 0.0:
            break
        # Decide the side to repeat the steps
        if func(miu) * func(a) < 0:
            b = miu
        else:
            a = miu
        if (b - a) <= epsilon:
            break
    return miu


def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1, 1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1, 1) + 1e-15)
    return (1 - (t1 * t2).sum(1)).mean()


def to_symmetric(edge_index, edge_weight, n, op="mean"):
    symmetric_edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index, symmetric_edge_weight, m=n, n=n, op=op
    )
    return symmetric_edge_index, symmetric_edge_weight
