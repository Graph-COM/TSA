import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.utils import edgeidx_to_adj, to_boolean_mask


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

        graph.src_mask = to_boolean_mask(idx, graph.num_nodes)
        graph.tgt_mask = to_boolean_mask(idx, graph.num_nodes)

        graph.edge_weight = torch.ones(graph.num_edges)
        return graph
