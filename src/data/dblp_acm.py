import os
import pdb
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array

from src.utils import to_boolean_mask


class DBLP_ACM(InMemoryDataset):
    def __init__(self, root, name, data_config):
        self.root = Path(root)
        self.name = name
        self.seed = data_config.seed
        self.target_val = data_config.target_val
        super().__init__(self.root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        file_name = f"data_{self.name}_{self.target_val}_{self.seed}.pt"
        return [str(file_name)]

    def process(self):
        print("processing")
        data = self.dblp_acm_generate()
        self.save([data], self.processed_paths[0])

    def read_files(self, path):
        path = Path(path)  # Ensure path is a Path object
        return path.read_text().splitlines()

    def dblp_acm_generate(self) -> Data:
        docs_path = Path(self.root) / self.name / f"raw/{self.name}_docs.txt"
        x = []
        for line in self.read_files(docs_path):
            x.append(line.split(","))
        x = np.array(x, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        edge_path = Path(self.root) / self.name / f"raw/{self.name}_edgelist.txt"
        edge_index = read_txt_array(edge_path, sep=",", dtype=torch.long).t()
        label_path = Path(self.root) / self.name / f"raw/{self.name}_labels.txt"
        y = []
        for line in self.read_files(label_path):
            y.append(line.replace("\r", "").replace("\n", ""))
        y = np.array(y, dtype=int)

        num_class = np.unique(y)
        class_index = []
        for i in num_class:
            c_i = np.where(y == i)[0]
            class_index.append(c_i)
        src_train_mask = np.array([])
        src_val_mask = np.array([])
        src_test_mask = np.array([])
        tgt_val_mask = np.array([])
        tgt_test_mask = np.array([])
        for idx in class_index:
            np.random.shuffle(idx)
            src_train_mask = np.concatenate(
                (src_train_mask, idx[0 : int(len(idx) * 0.6)]), 0
            )
            src_val_mask = np.concatenate(
                (src_val_mask, idx[int(len(idx) * 0.6) : int(len(idx) * 0.8)]), 0
            )
            src_test_mask = np.concatenate(
                (src_test_mask, idx[int(len(idx) * 0.8) :]), 0
            )
            tgt_val_mask = np.concatenate(
                (tgt_val_mask, idx[0 : int(len(idx) * self.target_val)]), 0
            )
            tgt_test_mask = np.concatenate(
                (tgt_test_mask, idx[int(len(idx) * self.target_val) :]), 0
            )
        src_train_mask = src_train_mask.astype(int)
        src_val_mask = src_val_mask.astype(int)
        src_test_mask = src_test_mask.astype(int)
        tgt_val_mask = tgt_val_mask.astype(int)
        tgt_test_mask = tgt_test_mask.astype(int)

        y = torch.from_numpy(y).to(torch.int64)
        graph = Data(edge_index=edge_index, x=x, y=y)
        graph.num_classes = torch.max(graph.y) + 1
        graph.num_nodes = graph.x.size(0)
        graph.src_train_mask = to_boolean_mask(src_train_mask, graph.num_nodes)
        graph.src_val_mask = to_boolean_mask(src_val_mask, graph.num_nodes)
        graph.src_test_mask = to_boolean_mask(src_test_mask, graph.num_nodes)
        graph.tgt_val_mask = to_boolean_mask(tgt_val_mask, graph.num_nodes)
        graph.tgt_test_mask = to_boolean_mask(tgt_test_mask, graph.num_nodes)
        src_mask = to_boolean_mask(
            np.concatenate((src_train_mask, src_val_mask, src_test_mask), 0),
            graph.num_nodes,
        )
        tgt_mask = to_boolean_mask(
            np.concatenate((tgt_val_mask, tgt_test_mask), 0), graph.num_nodes
        )
        graph.src_mask = src_mask
        graph.tgt_mask = tgt_mask
        graph.edge_weight = torch.ones(graph.num_edges)
        return graph
