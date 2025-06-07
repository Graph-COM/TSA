from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import BatchNorm1d, Linear, Parameter
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.utils import Metrics, SaveEmb

from .base_model import BaseModel, gcn_norm
from .model_manager import MODEL_REGISTRY


class GPR_prop(MessagePassing):
    """
    propagation class for GPR_GNN
    """

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr="add", **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ["SGC", "PPR", "NPPR", "Random", "WS"]
        if Init == "SGC":
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == "PPR":
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == "NPPR":
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "Random":
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == "WS":
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == "SGC":
            self.temp.data[self.alpha] = 1.0
        elif self.Init == "PPR":
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == "NPPR":
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == "Random":
            bound = np.sqrt(3 / (self.K + 1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == "WS":
            self.temp.data = self.Gamma

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        hidden = x * (self.temp[0])
        for k in range(self.K):
            if hasattr(self, "_scaled_edge_weights"):
                edge_weight = self._scaled_edge_weights[k]
                new_edge_index, new_edge_weight = gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
                )
            else:
                new_edge_index, new_edge_weight = gcn_norm(
                    edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
                )

            x = self.propagate(new_edge_index, x=x, norm=new_edge_weight)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return "{}(K={}, temp={})".format(self.__class__.__name__, self.K, self.temp)


@MODEL_REGISTRY.register()
class GPRGNN(BaseModel):
    def __init__(self, metrics: Metrics, args: DictConfig):
        super().__init__(metrics, args)
        self.lin1 = Linear(args.input_dim, args.gnn_dim)
        self.lin2 = Linear(args.gnn_dim, args.gnn_dim)
        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.bn_feature = args.bn_feature
        self.bn_classifier = args.bn_classifier
        self.prop1 = GPR_prop(args.K, args.alpha, args.Init, None)

        # print(list(self.prop1.named_parameters()))

        if self.bn_feature:
            self.bn1 = BatchNorm1d(args.gnn_dim)
        if self.bn_classifier:
            self.bn_mlp = nn.BatchNorm1d(args.cls_dim)

        # classification layer
        self.mlp_classify = nn.ModuleList()
        if args.cls_layers == 1:
            self.mlp_classify.append(nn.Linear(args.gnn_dim, args.num_classes))
        else:
            self.mlp_classify.append(nn.Linear(args.gnn_dim, args.cls_dim))
            for i in range(args.cls_layers - 2):
                self.mlp_classify.append(nn.Linear(args.cls_dim, args.cls_dim))
            self.mlp_classify.append(nn.Linear(args.cls_dim, args.num_classes))

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.bn_feature:
            x = F.relu(self.bn1(self.lin1(x)))
        else:
            x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # Propagate
        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)

        y = x
        for i in range(len(self.mlp_classify)):
            y = self.mlp_classify[i](y)
            if i != (len(self.mlp_classify) - 1):
                if self.bn_classifier:
                    y = self.bn_mlp(y)
                y = F.relu(y)

        return x, y

    def _custom_src_stats(self, data: Data):
        chosen_layers = []
        for m in self.modules():
            if isinstance(m, GPR_prop):
                chosen_layers.append(m)

        n_chosen_layers = len(chosen_layers)
        hook_list = [SaveEmb() for _ in chosen_layers]
        clean_mean = []
        clean_var = []

        hooks = [
            chosen_layers[i].register_forward_hook(hook_list[i])
            for i in range(n_chosen_layers)
        ]
        with torch.no_grad():
            self.eval()
            _ = self(data)
            for yy in range(n_chosen_layers):
                hook_list[yy].statistics_update(), hook_list[yy].clear(), hooks[
                    yy
                ].remove()

        for i in range(n_chosen_layers):
            clean_mean.append(hook_list[i].pop_mean()), clean_var.append(
                hook_list[i].pop_var()
            )

        return {
            "src_feat_mean": clean_mean,
            "src_feat_var": clean_var,
        }
