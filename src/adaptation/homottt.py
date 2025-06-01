import logging
import pdb
import time
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import KMeans

import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.utils import homophily, scatter, is_undirected
from torch_sparse import coalesce

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class HomoTTT(BaseAdapter):
    """
    Our proposed method based on Laplacian Regularization.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)
        self.epochs = adapter_config.epochs
        self.learning_rate = adapter_config.lr
        self.k0 = adapter_config.k0

    def adapt(self, data: Data) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)
        data = data.to(self.device)
        self.data = data

        # === Clone the original model before adaptation ===
        self.original_model = copy.deepcopy(self.model)
        self.original_model.eval().to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.data = data
        feat = data.x
        edge_index, edge_weight = data.edge_index, data.edge_weight

        for it in range(self.epochs):
            optimizer.zero_grad()
            loss = self.test_time_loss(feat, edge_index, edge_weight)
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            # Get outputs from original and updated models
            original_logits = self.original_model(data)[1]
            updated_logits = self.model(data)[1]

            original_probs = F.softmax(original_logits, dim=-1)  # shape: (N, C)
            updated_probs = F.softmax(updated_logits, dim=-1)

            # Get pseudo-labels from the emsemble of two models 
            # Gives slightly better results than updated_probs
            ensemble_probs = 0.5 * (original_probs + updated_probs)
            pseudo_labels = torch.argmax(ensemble_probs, dim=1)

            final_probs = select_model_predictions(
                original_probs=original_probs,
                updated_probs=updated_probs,
                edge_index=data.edge_index,
                pseudo_labels=pseudo_labels,
                k0=self.k0
            )

        return final_probs

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
            p=0.5,
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
            with torch.no_grad():
                Z, _ = self.model(self.data)
                kmeans = KMeans(n_clusters=self.data.num_classes, n_init=10)
                # KMeans from sklearn too slow
                pseudo_labels, _ = kmeans_torch(Z, num_clusters=self.data.num_classes)
            edge_index, edge_mask = homophily_based_dropout_edge(edge_index=edge_index, pseudo_labels=pseudo_labels, drop_prob=p, force_undirected=True)
            edge_weight = edge_weight[edge_mask]
            # edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)

        Z, _ = self.augument_predict(feat, edge_index, edge_weight)
        return Z

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1, 1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1, 1) + 1e-15)
    return (1 - (t1 * t2).sum(1)).mean()

def homophily_based_dropout_edge(edge_index: Tensor,
                                  pseudo_labels: Tensor,
                                  drop_prob: float = 0.5,
                                  force_undirected: bool = False,
                                  training: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Drop edges with probability inversely related to homophily score.
    Higher homophily -> lower drop probability.
    """
    if not training or drop_prob == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    num_nodes = int(edge_index.max()) + 1  # assuming zero-based indexing
    node_homophily = compute_node_homophily(edge_index, pseudo_labels, num_nodes)
    row, col = edge_index
    edge_homophily = (node_homophily[row] + node_homophily[col]) / 2.

    # Normalize homophily to drop probabilities: low homophily -> high drop probability
    drop_probs = drop_prob * (1.0 - edge_homophily)

    # Sample using edge-wise dropout probability
    edge_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_probs

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def compute_node_homophily(edge_index: Tensor, pseudo_labels: Tensor, num_nodes: int) -> Tensor:
    """
    Compute homophily scores for each edge.
    """
    row, col = edge_index
    out = torch.zeros(row.size(0), device=row.device)
    out[pseudo_labels[row] == pseudo_labels[col]] = 1.
    out = scatter(out, col, 0, dim_size=pseudo_labels.size(0), reduce='mean')
    return out

def kmeans_torch(x, num_clusters, num_iters=10):
    """
    x: (N, D) input tensor
    Returns: cluster_labels (N,), cluster_centers (num_clusters, D)
    """
    N, D = x.size()
    # Randomly initialize cluster centers
    indices = torch.randperm(N)[:num_clusters]
    centroids = x[indices]

    for _ in range(num_iters):
        # Compute distances: (N, num_clusters)
        dist = torch.cdist(x, centroids, p=2)

        # Assign points to nearest cluster
        labels = dist.argmin(dim=1)

        # Recompute centroids
        centroids = torch.stack([
            x[labels == i].mean(dim=0) if (labels == i).any() else centroids[i]
            for i in range(num_clusters)
        ])

    return labels, centroids

def select_model_predictions(original_probs: torch.Tensor, updated_probs: torch.Tensor,
                             edge_index: torch.Tensor, pseudo_labels: torch.Tensor,
                             k0: float = 1.0) -> torch.Tensor:
    """
    Select between original or updated model probabilities based on node-wise homophily scores.
    
    Args:
        original_probs: (N, C) softmax probabilities from original model
        updated_probs: (N, C) softmax probabilities from updated model
        edge_index: (2, E) graph edge index
        pseudo_labels: (N,) pseudo labels (from KMeans or ensemble)
        k0: sigmoid sharpness for acceptance
    
    Returns:
        (N, C) final probabilities for each node
    """
    num_nodes = original_probs.size(0)
    node_homophily = compute_node_homophily(edge_index, pseudo_labels, num_nodes)

    # Acceptance probabilities (per node)
    acceptance_probs = torch.sigmoid(k0 * node_homophily)

    # Sample binary acceptance decision per node
    acceptance = torch.bernoulli(acceptance_probs)
    
    # Select updated or original probabilities based on acceptance
    final_probs = torch.where(
        acceptance.unsqueeze(1).bool(),
        updated_probs,
        original_probs
    )

    return final_probs
