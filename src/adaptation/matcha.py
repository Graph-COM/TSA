import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch_geometric.data import Data

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter
from .lame import LAME
from .t3a import T3A
from .tent import TENT


@ADAPTER_REGISTRY.register()
class Matcha(BaseAdapter):
    """
    
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)
        self.adapter_config = adapter_config
        self.source_stats = source_stats
        self.iter_epochs = adapter_config.iter_epochs
        self.ada_lr = adapter_config.ada_lr
        self.base_tta = adapter_config.base_tta

    def partial_freeze(self):
        self.model.requires_grad_(False)
        self.model.prop1.temp.requires_grad_(True)

    def initialize_base_tta(self, help_model):
        if self.base_tta == "TENT":
            return TENT(help_model, self.source_stats, self.adapter_config)
        elif self.base_tta == "LAME":
            return LAME(help_model, self.source_stats, self.adapter_config)
        elif self.base_tta == "T3A":
            return T3A(help_model, self.source_stats, self.adapter_config)
        else:
            raise ValueError(f"Unknown base_tta method: {self.base_tta}")

    def base_TTA_adapt(self, data: Data):
        if self.base_tta == "ERM":
            self.model.eval()
            Z, output = self.model(data)
            probs = F.softmax(output, dim=-1)
        else:
            model_help = deepcopy(self.model)
            base_TTA = self.initialize_base_tta(model_help)
            probs = base_TTA.adapt(data)
            del base_TTA
            Z, _ = self.model(data)
        return Z, probs

    def adapt(self, data: Data) -> torch.Tensor:
        self.model.to(self.device)
        data = data.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.ada_lr)

        for epoch in range(self.iter_epochs):
            optimizer.zero_grad()

            Z, probs = self.base_TTA_adapt(data)
            probs = probs.detach()

            self.partial_freeze()
            loss = pic_loss(Z, probs)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(probs, 1)
                accuracy = (predicted == data.y).float().mean().item()
                labels = data.y.cpu().numpy()
                predicted = predicted.cpu().numpy()
                bal_accuracy = balanced_accuracy_score(labels, predicted)

            print(
                f"Epoch {epoch+1}, Loss: {loss.item():.5f}, Accuracy: {accuracy:.5f}, Bal Accuracy: {bal_accuracy:.5f}"
            )

        return probs


def pic_loss(feats, prob):
    _, c = prob.shape
    prob = prob.to(feats.device)

    mus = (prob.T @ feats) / prob.sum(dim=0).view(c, 1)  # weight average, c * k
    sq_dist = torch.square(torch.cdist(feats, mus, p=2))  # num_nodes * c
    var_intra = (sq_dist * prob).sum()
    var_total = torch.sum(torch.square(feats - feats.mean(dim=0)))
    return var_intra / var_total
