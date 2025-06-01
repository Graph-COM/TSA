import os
import pdb
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import Data, InMemoryDataset

from src.utils import edgeidx_to_adj, to_boolean_mask


def take_second(element):
    return element[1]


def load_ogb_arxiv(data_dir, start_year, end_year, proportion=1.0):
    start_year, end_year = int(start_year), int(end_year)
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=data_dir)
    graph = dataset.graph

    node_years = graph["node_year"]
    n = node_years.shape[0]
    node_years = node_years.reshape(n)

    d = np.zeros(len(node_years))

    edges = graph["edge_index"]
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= end_year and node_years[edges[1][i]] <= end_year:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= end_year:
            nodes.append([i, d[i]])

    nodes.sort(key=take_second, reverse=True)

    nodes = nodes[: int(proportion * len(nodes))]

    result_edges = []
    result_features = []
    result_labels = []

    for node in nodes:
        result_features.append(graph["node_feat"][node[0]])
    result_features = np.array(result_features)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    node_feat = torch.tensor(result_features, dtype=torch.float)
    dataset.graph = {
        "edge_index": edge_index,
        "edge_feat": None,
        "node_feat": node_feat,
        "num_nodes": node_feat.size(0),
    }
    dataset.label = torch.tensor(result_labels)
    node_years_new = [node_years[node[0]] for node in nodes]
    dataset.test_mask = torch.tensor(node_years_new) > start_year

    return dataset


class Arxiv(InMemoryDataset):
    def __init__(self, root, years, data_config):
        self.root = Path(root)
        self.start, self.end = years.split("_")
        self.seed = data_config.seed
        self.target_val = data_config.target_val
        assert self.target_val == 0.03
        super().__init__(self.root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        file_name = (
            f"data_{self.start}_{self.end}_{self.target_val}_{self.seed}.pt"
        )
        return [str(file_name)]

    def process(self):
        print("processing")
        data = self.arxiv_generate()
        self.save([data], self.processed_paths[0])

    def arxiv_generate(self) -> Data:
        dataset = load_ogb_arxiv(self.root.parent, self.start, self.end)
        adj = edgeidx_to_adj(
            dataset.graph["edge_index"][0],
            dataset.graph["edge_index"][1],
            dataset.graph["num_nodes"],
        )
        edge_index = torch.from_numpy(
            np.array([adj.nonzero()[0], adj.nonzero()[1]])
        ).long()
        graph = Data(
            edge_index=edge_index,
            x=dataset.graph["node_feat"],
            y=dataset.label.view(-1),
        )

        graph.num_classes = torch.max(graph.y) + 1
        graph.num_nodes = graph.x.size(0)

        idx = (dataset.test_mask == True).nonzero().view(-1).numpy()
        np.random.shuffle(idx)
        idx_len = idx.shape[0]
        graph.src_train_mask = to_boolean_mask(
            idx[0 : int(0.6 * idx_len)], graph.num_nodes
        )
        graph.src_val_mask = to_boolean_mask(
            idx[int(0.6 * idx_len) : int(0.8 * idx_len)], graph.num_nodes
        )
        graph.src_test_mask = to_boolean_mask(
            idx[int(0.8 * idx_len) :], graph.num_nodes
        )
        graph.tgt_val_mask = to_boolean_mask(
            idx[0 : int(self.target_val * idx_len)], graph.num_nodes
        )
        graph.tgt_test_mask = to_boolean_mask(
            idx[int(self.target_val * idx_len) :], graph.num_nodes
        )

        # tgt_val_mask, tgt_test_mask = num_per_class_split(
        #     idx, graph, self.target_val
        # )
        # graph.tgt_val_mask = to_boolean_mask(tgt_val_mask, graph.num_nodes)
        # graph.tgt_test_mask = to_boolean_mask(tgt_test_mask, graph.num_nodes)
        # graph.target_validation_mask = idx[0 : int(0.2 * idx_len)]
        # graph.target_testing_mask = idx[int(0.2 * idx_len) :]
        graph.src_mask = to_boolean_mask(idx, graph.num_nodes)
        graph.tgt_mask = to_boolean_mask(idx, graph.num_nodes)

        graph.edge_weight = torch.ones(graph.num_edges)
        return graph
