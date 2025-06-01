import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.utils import edgeidx_to_adj, to_boolean_mask


class Syn_MAG(InMemoryDataset):
    def __init__(self, root, lang, data_config):
        self.root = root
        self.lang = lang
        self.seed = data_config.seed
        self.target_val = data_config.target_val
        assert self.target_val == 0.03
        self.data_config = data_config
        self.datadir = f"./data/MAG/raw/{self.lang}_labels_20.pt"
        print(self.datadir)
        super().__init__(root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        file_name = f"data_syn_{self.lang}_{self.target_val}_{self.seed}.pt"
        return [str(file_name)]

    def process(self):
        print("processing")
        data = self.mag_generate()
        self.save([data], self.processed_paths[0])

    def mag_generate(self) -> Data:
        graph = torch.load(self.datadir)
        adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
        # graph.adj = adj
        graph.edge_index = torch.from_numpy(
            np.array([adj.nonzero()[0], adj.nonzero()[1]])
        ).long()
        graph.num_classes = torch.max(graph.y) + 1
        graph.num_nodes = graph.num_nodes

        # generate synthetic features
        features = synthetic_features(graph, self.data_config.num_classes)
        graph.x = features

        idx = np.arange(graph.num_nodes)
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
        # graph.target_validation_mask = idx[0 : int(0.2 * idx_len)]
        # graph.target_testing_mask = idx[int(0.2 * idx_len) :]
        graph.src_mask = to_boolean_mask(idx, graph.num_nodes)
        graph.tgt_mask = to_boolean_mask(idx, graph.num_nodes)

        graph.edge_weight = torch.ones(graph.num_edges)

        return graph


def synthetic_features(graph, num_classes, sigma=0.3):
    MU = np.eye(num_classes)
    features = np.zeros((graph.num_nodes, MU.shape[1]))

    for i in range(num_classes):
        class_indices = (graph.y == i).nonzero(as_tuple=True)[0]
        class_size = len(class_indices)
        MU_i = MU[i]
        C = np.random.multivariate_normal(
            mean=MU_i, cov=np.eye(MU.shape[1]) * sigma**2, size=class_size
        )
        features[class_indices] = C

    features = torch.FloatTensor(features)
    return features


class MAG(InMemoryDataset):
    def __init__(self, root, lang, data_config):
        self.root = root
        self.lang = lang
        self.seed = data_config.seed
        self.target_val = data_config.target_val
        self.datadir = f"./data/MAG/raw/{self.lang}_labels_20.pt"
        print(self.datadir)
        super().__init__(root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        file_name = f"data_{self.lang}_{self.target_val}_{self.seed}.pt"
        return [str(file_name)]

    def process(self):
        print("processing")
        data = self.mag_generate()
        self.save([data], self.processed_paths[0])

    def mag_generate(self) -> Data:
        graph = torch.load(self.datadir)
        adj = edgeidx_to_adj(graph.edge_index[0], graph.edge_index[1], graph.num_nodes)
        # graph.adj = adj
        graph.edge_index = torch.from_numpy(
            np.array([adj.nonzero()[0], adj.nonzero()[1]])
        ).long()
        graph.num_classes = torch.max(graph.y) + 1
        graph.num_nodes = graph.x.size(0)

        idx = np.arange(graph.num_nodes)
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
        # graph.target_validation_mask = idx[0 : int(0.2 * idx_len)]
        # graph.target_testing_mask = idx[int(0.2 * idx_len) :]
        graph.src_mask = to_boolean_mask(idx, graph.num_nodes)
        graph.tgt_mask = to_boolean_mask(idx, graph.num_nodes)

        graph.edge_weight = torch.ones(graph.num_edges)
        return graph
