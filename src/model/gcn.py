from typing import List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.utils import Metrics, SaveEmb

from .base_model import BaseModel, gcn_norm
from .model_manager import MODEL_REGISTRY


class GCN_reweight(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        num_nodes = x.size(0)
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, num_nodes=num_nodes, dtype=x.dtype
        )
        x = self.lin(x)

        out = self.propagate(
            edge_index,
            size=(num_nodes, num_nodes),
            x=x,
            edge_weight=edge_weight,
        )
        out = out + self.bias
        return out

    def message(self, x_j, edge_index, edge_weight):
        x_j = edge_weight.view(-1, 1) * x_j
        return x_j


@MODEL_REGISTRY.register()
class GCN(BaseModel):
    def __init__(self, metrics: Metrics, args: DictConfig):
        super().__init__(metrics, args)
        input_dim, output_dim = args.input_dim, args.num_classes

        self.dropout = args.dropout
        self.bn_feature = args.bn_feature
        self.bn_classifier = args.bn_classifier

        # conv layers
        self.conv = nn.ModuleList()
        self.conv.append(GCN_reweight(input_dim, args.gnn_dim))
        for i in range(args.gnn_layers - 1):
            self.conv.append(GCN_reweight(args.gnn_dim, args.gnn_dim))

        # bn layers
        if self.bn_feature:
            self.bns = nn.ModuleList()
            for i in range(args.gnn_layers - 1):
                self.bns.append(nn.BatchNorm1d(args.gnn_dim))
        if self.bn_classifier:
            self.bn_mlp = nn.BatchNorm1d(args.cls_dim)

        # classification layer
        self.mlp_classify = nn.ModuleList()
        if args.cls_layers == 1:
            self.mlp_classify.append(nn.Linear(args.gnn_dim, output_dim))
        else:
            self.mlp_classify.append(nn.Linear(args.gnn_dim, args.cls_dim))
            for i in range(args.cls_layers - 2):
                self.mlp_classify.append(nn.Linear(args.cls_dim, args.cls_dim))
            self.mlp_classify.append(nn.Linear(args.cls_dim, output_dim))

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for i, layer in enumerate(self.conv):
            x = layer(x, edge_index, edge_weight)

            if self.bn_feature and (i != len(self.conv) - 1):
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

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
            if isinstance(m, GCN_reweight):
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
