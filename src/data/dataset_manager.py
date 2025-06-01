from omegaconf import DictConfig
from torch_geometric.datasets import Planetoid

from .arxiv import Arxiv
from .csbm import CSBM
from .dblp_acm import DBLP_ACM
from .mag import MAG, Syn_MAG
from .pileup import Pileup


def dataset_manager(src_tgt: str, data_config: DictConfig):
    if src_tgt == "source":
        setting = data_config.source
    elif src_tgt == "target":
        setting = data_config.target
        
    if data_config.name == "CSBM":
        dataset = CSBM(
            root=data_config.root,
            setting=setting,
            data_config=data_config,
        )
    elif data_config.name == "Pileup":
        signal = setting.split("_")[0]
        pu = setting.split("_")[1]
        dataset = Pileup(
            root=data_config.root, signal=signal, pu=pu, data_config=data_config
        )
    elif data_config.name == "Arxiv":
        dataset = Arxiv(
            root=data_config.root,
            years=setting,
            data_config=data_config,
        )
    elif data_config.name == "DBLP_ACM":
        dataset = DBLP_ACM(
            root=data_config.root,
            name=setting,
            data_config=data_config,
        )
    elif data_config.name == "MAG":
        if data_config.synthetic:
            dataset = Syn_MAG(
                root=data_config.root,
                lang=setting,
                data_config=data_config,
            )
        else:
            dataset = MAG(
                root=data_config.root,
                lang=setting,
                data_config=data_config,
            )
        # import torch
        # class_19_nodes = (dataset.y == 19).nonzero(as_tuple=True)[0]
        # edge_index = dataset.edge_index
        # # mask = ~((edge_index[0].unsqueeze(1) == class_19_nodes).any(dim=1) |
        # #  (edge_index[1].unsqueeze(1) == class_19_nodes).any(dim=1))
        # node_mask = torch.ones(dataset.num_nodes, dtype=torch.bool, device=dataset.x.device)
        # node_mask[class_19_nodes] = False

        # # Map old node indices to new node indices
        # node_idx_map = torch.full((node_mask.sum() + len(class_19_nodes),), -1, dtype=torch.long, device=dataset.x.device)
        # node_idx_map[node_mask.nonzero(as_tuple=True)[0]] = torch.arange(node_mask.sum(), device=dataset.x.device)

        # # Update edge indices
        # new_edge_index = node_idx_map[dataset.edge_index]
        # valid_edge_mask = (new_edge_index[0] >= 0) & (new_edge_index[1] >= 0)
        # new_edge_index = new_edge_index[:, valid_edge_mask]

        # # Remove edges with degree 0 nodes
        # from torch_geometric.utils import degree
        # node_degrees = degree(new_edge_index[0], num_nodes=node_mask.sum())
        # degree_0_nodes = (node_degrees == 0).nonzero(as_tuple=True)[0]

        # if len(degree_0_nodes) > 0:
        #     # Update the node mask to exclude degree 0 nodes
        #     final_node_mask = torch.ones(node_mask.sum(), dtype=torch.bool, device=dataset.x.device)
        #     final_node_mask[degree_0_nodes] = False

        #     # Update node index mapping for the remaining nodes
        #     final_node_idx_map = torch.full((final_node_mask.sum() + len(degree_0_nodes),), -1, dtype=torch.long, device=dataset.x.device)
        #     final_node_idx_map[final_node_mask.nonzero(as_tuple=True)[0]] = torch.arange(final_node_mask.sum(), device=dataset.x.device)

        #     # Update edge index again
        #     final_new_edge_index = final_node_idx_map[new_edge_index]
        #     final_valid_edge_mask = (final_new_edge_index[0] >= 0) & (final_new_edge_index[1] >= 0)
        #     final_new_edge_index = final_new_edge_index[:, final_valid_edge_mask]

        #     # Update node features and labels
        #     final_new_x = dataset.x[node_mask][final_node_mask]
        #     final_new_y = dataset.y[node_mask][final_node_mask]
        #     final_new_edge_weight = dataset.edge_weight[valid_edge_mask][final_valid_edge_mask]

        #     # Return or use the final_new_edge_index, final_new_x, final_new_y, final_new_edge_weight as needed

        # else:
        #     # If no degree 0 nodes are found, use the previous results
        #     final_new_edge_index = new_edge_index
        #     final_new_edge_weight = dataset.edge_weight[valid_edge_mask]
        #     final_new_x = dataset.x[node_mask]
        #     final_new_y = dataset.y[node_mask]

        # # import pdb
        # # pdb.set_trace()
        # # assert torch.all(valid_edge_mask)

        # from torch_geometric.data import Data
        # dataset_n = Data(x=final_new_x, edge_index=final_new_edge_index, y=final_new_y, edge_weight=final_new_edge_weight)
        # dataset_n.num_classes=19
        # dataset_n.num_nodes = final_new_x.size(0)
        # dataset_n.src_train_mask = dataset.src_train_mask[node_mask][final_node_mask]
        # dataset_n.src_val_mask = dataset.src_val_mask[node_mask][final_node_mask]
        # dataset_n.src_test_mask = dataset.src_test_mask[node_mask][final_node_mask]
        # dataset_n.tgt_val_mask = dataset.tgt_val_mask[node_mask][final_node_mask]
        # dataset_n.tgt_test_mask = dataset.tgt_test_mask[node_mask][final_node_mask]
        # dataset_n.src_mask = dataset.src_mask[node_mask][final_node_mask]
        # dataset_n.tgt_mask = dataset.tgt_mask[node_mask][final_node_mask]
        # # import pdb
        # # dataset.edge_index = new_edge_index
        # # pdb.set_trace()
        # dataset = dataset_n
    elif data_config.name == "Cora":
        dataset = Planetoid(
            root=data_config.root,
            name="Cora",
        )
    else:
        raise ValueError(f"Dataset {data_config.name} not supported")

    return dataset
