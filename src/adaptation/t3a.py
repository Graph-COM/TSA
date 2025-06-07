import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data

from .adapter_manager import ADAPTER_REGISTRY
from .base_adapter import BaseAdapter


@ADAPTER_REGISTRY.register()
class T3A(BaseAdapter):
    """
    Our proposed method based on Laplacian Regularization.
    """

    def __init__(self, pre_model, source_stats, adapter_config):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(pre_model, source_stats, adapter_config)
        self.filter_K = adapter_config.filter_K

        self.classifier = pre_model.mlp_classify[-1]
        self.num_classes = pre_model.mlp_classify[-1].out_features
        self.featurizer = get_gnn_featurer(pre_model)

        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=self.num_classes
        ).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

    @torch.no_grad()
    def adapt(self, data: Data) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)
        data = data.to(self.device)
        _, z = self.featurizer(data)

        # # online adaptation
        p = self.classifier(z)
        yhat = torch.nn.functional.one_hot(
            p.argmax(1), num_classes=self.num_classes
        ).float()
        ent = softmax_entropy(p)

        # prediction
        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = supports.T @ (labels)
        logits = z @ torch.nn.functional.normalize(weights, dim=0)
        return F.softmax(logits, dim=1)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).to(self.device)

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(self.device)
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            # print(indices1[y_hat == i][indices2][:filter_K])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_gnn_featurer(model):
    model.mlp_classify[-1] = nn.Identity()
    return model
