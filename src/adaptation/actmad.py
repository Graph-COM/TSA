import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from src.model import GSN, GPRGNN, GCN
from src.utils import SaveEmb

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class ActMAD(BaseAdapter):
    """
    ActMAD from "ActMAD: Activation Matching to Align Distributions 
    for Test-Time Training (CVPR 2023)"
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        super().__init__(pre_model, source_stats, adapter_config)
        self.epochs = adapter_config.epochs
        self.learning_rate = adapter_config.lr
        self.weight_decay = adapter_config.weight_decay
        self.source_stats = source_stats

        if isinstance(self.model, GSN):
            self.align_module = type(self.model.conv[0])
        elif isinstance(self.model, GPRGNN):
            self.align_module = type(self.model.prop1)
        elif isinstance(self.model, GCN):
            self.align_module = type(self.model.conv[0])

    def adapt(self, data: Data) -> torch.Tensor:
        self.model.to(self.device)
        data = data.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        l1_loss = nn.L1Loss(reduction='mean')
        src_feat_mean = [mean.to(self.device) for mean in self.source_stats["src_feat_mean"]]
        src_feat_var = [var.to(self.device) for var in self.source_stats["src_feat_var"]]
        n_chosen_layers = len(src_feat_mean)

        self.model.train()
        chosen_layers = []
        for m in self.modules():
            if isinstance(m, self.align_module):
                chosen_layers.append(m)

        optimizer.zero_grad()
        save_outputs_tta = [SaveEmb() for _ in range(n_chosen_layers)]
        hooks_list_tta = [chosen_layers[i].register_forward_hook(save_outputs_tta[i])
                            for i in range(n_chosen_layers)]

        _, output = self.model(data)
        act_mean_batch_tta = []
        act_var_batch_tta = []
        for yy in range(n_chosen_layers):
            save_outputs_tta[yy].statistics_update()
            act_mean_batch_tta.append(save_outputs_tta[yy].pop_mean())
            act_var_batch_tta.append(save_outputs_tta[yy].pop_var())

        for z in range(n_chosen_layers):
            save_outputs_tta[z].clear()
            hooks_list_tta[z].remove()

        loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
        loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().cuda()
        for i in range(n_chosen_layers):
            loss_mean += l1_loss(act_mean_batch_tta[i].cuda(), src_feat_mean[i].cuda())
            loss_var += l1_loss(act_var_batch_tta[i].cuda(), src_feat_var[i].cuda())
        loss = (loss_mean + loss_var) * 0.5

        loss.backward()
        optimizer.step()

        self.model.eval()
        _, output = self.model(data)
        probs = F.softmax(output, dim=-1)
        return probs


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
