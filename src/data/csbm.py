import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.utils import to_boolean_mask


def _edges_to_adj(edges, num_node):
    edge_source = [int(i[0]) for i in edges]
    edge_target = [int(i[1]) for i in edges]
    data = np.ones(len(edge_source))
    adj = sp.csr_matrix((data, (edge_source, edge_target)), shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    rows, columns = adj.nonzero()
    edge_index = torch.tensor([rows, columns], dtype=torch.long)
    return adj, edge_index


class CSBM(InMemoryDataset):
    def __init__(self, root, setting, data_config):
        self.root = root
        self.setting = setting
        self.sigma = data_config.sigma
        # if seed is not None:
        #     self.seed = seed
        # else:
        self.seed = data_config.seed
        self.target_val = data_config.target_val
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        file_name = f"data_{self.setting}_{self.target_val}_{self.seed}.pt"
        return [str(file_name)]

    def process(self):
        data = self.CSBM_generate()
        self.save([data], self.processed_paths[0])

    def CSBM_generate(self) -> Data:
        d = 3
        num_nodes = 6000
        MU = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if self.setting == "src":
            py = [1 / 3, 1 / 3, 1 / 3]
            B = [[0.02, 0.005, 0.005], [0.005, 0.02, 0.005], [0.005,  0.005, 0.02]]
        elif self.setting == "src_imb":
            py = [0.1, 0.3, 0.6]
            B = [[0.02, 0.005, 0.005], [0.005, 0.02, 0.005], [0.005,  0.005, 0.02]]
        elif self.setting == "nbr1":
            py = [0.1, 0.3, 0.6]
            B = [[0.01, 0.0075, 0.0075], [0.0075,  0.01,  0.0075], [0.0075,  0.0075, 0.01]]
        elif self.setting == "nbr2":
            py = [0.1, 0.3, 0.6]
            B = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
        elif self.setting == "css1":
            py = [0.1, 0.3, 0.6]
            B = [[0.01, 0.0075, 0.0075], [0.0075,  0.01,  0.0075], [0.0075,  0.0075, 0.01]]
            B = [[i/2 for i in row] for row in B]
        elif self.setting == "css2":
            py = [0.1, 0.3, 0.6]
            B = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
            B = [[i/2 for i in row] for row in B]
        elif self.setting == "str1":
            py = [0.1, 0.3, 0.6]
            # B = [[0.075, 0.025, 0.025], [0.025, 0.075, 0.025], [0.025, 0.025, 0.075]]
            B = [[0.015, 0.0075, 0.0075], [0.0075,  0.015,  0.0075], [0.0075,  0.0075, 0.015]]
            B = [[i/2 for i in row] for row in B]
        elif self.setting == "str2":
            py = [0.1, 0.3, 0.6]
            B = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
            B = [[i/2 for i in row] for row in B]
        elif self.setting == "str3":
            py = [1 / 3, 1 / 3, 1 / 3]
            # B = [[0.075, 0.025, 0.025], [0.025, 0.075, 0.025], [0.025, 0.025, 0.075]]
            B = [[0.015, 0.0075, 0.0075], [0.0075,  0.015,  0.0075], [0.0075,  0.0075, 0.015]]
            B = [[i/2 for i in row] for row in B]
        elif self.setting == "str4":
            py = [1 / 3, 1 / 3, 1 / 3]
            B = [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
            B = [[i/2 for i in row] for row in B]

        N = [int(num_nodes * i) for i in py]
        B = [[i * 0.5 for i in row] for row in B]

        G = nx.stochastic_block_model(N, B)
        edge_list = list(G.edges)

        MU_0 = MU[0]
        MU_1 = MU[1]
        MU_2 = MU[2]
        C0 = np.random.multivariate_normal(
            mean=MU_0, cov=np.eye(d) * self.sigma**2, size=N[0]
        )
        C1 = np.random.multivariate_normal(
            mean=MU_1, cov=np.eye(d) * self.sigma**2, size=N[1]
        )
        C2 = np.random.multivariate_normal(
            mean=MU_2, cov=np.eye(d) * self.sigma**2, size=N[2]
        )

        num_nodes = np.sum(N)
        print(num_nodes)
        node_idx = np.arange(num_nodes)
        features = np.zeros((num_nodes, C1.shape[1]))
        label = np.zeros((num_nodes))

        c0_idx = node_idx[list(G.graph["partition"][0])]
        c1_idx = node_idx[list(G.graph["partition"][1])]
        c2_idx = node_idx[list(G.graph["partition"][2])]

        features[c0_idx] = C0
        features[c1_idx] = C1
        features[c2_idx] = C2

        label[c1_idx] = 1
        label[c2_idx] = 2

        random.shuffle(c0_idx)
        random.shuffle(c1_idx)
        random.shuffle(c2_idx)

        features = torch.FloatTensor(features)
        label = torch.LongTensor(label)
        num_nodes = len(label)
        adj, edge_index = _edges_to_adj(edge_list, num_nodes)
        graph = Data(x=features, edge_index=edge_index, y=label)
        graph.num_nodes = num_nodes

        idx_source_train = np.concatenate(
            (
                c0_idx[: int(0.6 * len(c0_idx))],
                c1_idx[: int(0.6 * len(c1_idx))],
                c2_idx[: int(0.6 * len(c2_idx))],
            )
        )
        idx_source_valid = np.concatenate(
            (
                c0_idx[int(0.6 * len(c0_idx)) : int(0.8 * len(c0_idx))],
                c1_idx[int(0.6 * len(c1_idx)) : int(0.8 * len(c1_idx))],
                c2_idx[int(0.6 * len(c2_idx)) : int(0.8 * len(c2_idx))],
            )
        )
        idx_source_test = np.concatenate(
            (
                c0_idx[int(0.8 * len(c0_idx)) :],
                c1_idx[int(0.8 * len(c1_idx)) :],
                c2_idx[int(0.8 * len(c2_idx)) :],
            )
        )
        idx_target_valid = np.concatenate(
            (
                c0_idx[: int(self.target_val * len(c0_idx))],
                c1_idx[: int(self.target_val * len(c1_idx))],
                c2_idx[: int(self.target_val * len(c2_idx))],
            )
        )
        idx_target_test = np.concatenate(
            (
                c0_idx[int(self.target_val * len(c0_idx)) :],
                c1_idx[int(self.target_val * len(c1_idx)) :],
                c2_idx[int(self.target_val * len(c2_idx)) :],
            )
        )
        graph.src_train_mask = to_boolean_mask(idx_source_train, graph.num_nodes)
        graph.src_val_mask = to_boolean_mask(idx_source_valid, graph.num_nodes)
        graph.src_test_mask = to_boolean_mask(idx_source_test, graph.num_nodes)
        graph.tgt_val_mask = to_boolean_mask(idx_target_valid, graph.num_nodes)
        graph.tgt_test_mask = to_boolean_mask(idx_target_test, graph.num_nodes)

        graph.src_mask = to_boolean_mask(np.arange(graph.num_nodes), graph.num_nodes)
        graph.tgt_mask = to_boolean_mask(np.arange(graph.num_nodes), graph.num_nodes)

        # graph.adj = adj
        graph.num_classes = 3
        graph.edge_weight = torch.ones(graph.num_edges)
        edge_class = np.zeros((graph.num_edges, graph.num_classes, graph.num_classes))
        for idx in range(graph.num_edges):
            i = graph.edge_index[0][idx]
            j = graph.edge_index[1][idx]
            edge_class[idx, graph.y[i], graph.y[j]] = 1
        graph.edge_class = edge_class
        print("done")
        return graph
