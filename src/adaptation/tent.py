import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch_geometric.data import Data

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class TENT(BaseAdapter):
    """
    TENT from "Tent: Fully Test-time Adaptation by
    Entropy Minimization (ICLR 2021)".
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        super().__init__(pre_model, source_stats, adapter_config)
        self.epochs = adapter_config.epochs
        self.learning_rate = adapter_config.lr
        self.configure_model()

        if self.adapter_config.bn_feature and self.adapter_config.bn_classifier:
            self.tent_param = list(self.model.bns.parameters()) + list(
                self.model.bn_mlp.parameters()
            )
        elif self.adapter_config.bn_classifier:
            self.tent_param = self.model.bn_mlp.parameters()
        elif self.adapter_config.bn_feature:
            self.tent_param = self.model.bns.parameters()
        else:
            raise ValueError(f"Do not contain batch normalization layers")

    def configure_model(self):
        self.model.train()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def adapt(self, data: Data) -> Tensor:
        self.model.to(self.device)
        data = data.to(self.device)
        optimizer = optim.Adam(self.tent_param, lr=self.learning_rate)
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            _, output = self.model(data)

            loss = softmax_entropy(output).mean(0)
            loss.backward()
            optimizer.step()

            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                _, output = self.model(data)
                val_loss = softmax_entropy(output).mean(0)
                _, predicted = torch.max(output, 1)

                val_accuracy = (
                    (predicted[data.tgt_val_mask] == data.y[data.tgt_val_mask])
                    .float()
                    .mean()
                    .item()
                )

        self.model.eval()
        _, output = self.model(data)
        probs = F.softmax(output, dim=-1)
        return probs


def softmax_entropy(x: Tensor) -> Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
