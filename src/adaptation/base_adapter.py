from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.data import Data


class BaseAdapter(nn.Module):
    def __init__(self, model, source_stats, adapter_config: DictConfig):
        super().__init__()
        self.adapter_config = adapter_config
        self.device = torch.device(
            f"{self.adapter_config.device}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.source_stats = source_stats

    def adaptation(self):
        raise NotImplementedError

    def run_in_batch(self, num_nodes: int, batch_size: int, func):
        """
        Run a function in batches. This is in particular for non-parametric methods
        that cannot fit into GPU memory.

        :param num_nodes: The number of nodes in the dataset.
        :param batch_size: The size of each batch.
        :param func: The function to be applied to each batch.
        """

        def wrapper(*args, **kwargs):
            """
            Wrapper function to handle batch processing.
            """
            num_batches = (num_nodes + batch_size - 1) // batch_size

            # Shuffle the indices
            indices = torch.randperm(num_nodes, device=self.device)
            # indices = torch.arange(num_nodes)

            # Shuffle all positional arguments
            shuffled_args = [arg[indices] for arg in args]

            outputs = []
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_nodes)
                batch_args = [arg[start:end] for arg in shuffled_args]

                # Perform the operation on the batch
                output = func(*batch_args, **kwargs)
                outputs.append(output)

            arg_1 = [arg[torch.argsort(indices)] for arg in shuffled_args]
            arg_2 = [arg for arg in args]
            # Assert that the shuffled arguments are the same as the original arguments
            for tsr1, tsr2 in zip(arg_1, arg_2):
                assert torch.equal(tsr1, tsr2), f"{tsr1} != {tsr2}"
            if isinstance(outputs[0], tuple):
                num_elements = len(outputs[0])
                pdb.set_trace()
                concatenated_outputs = [
                    torch.cat([output[i] for output in outputs])
                    for i in range(num_elements)
                ]
                pdb.set_trace()
                sorted_outputs = [
                    output[torch.argsort(indices)] for output in concatenated_outputs
                ]
                return tuple(sorted_outputs)
            else:
                # Concatenate the outputs and sort them back to the original order
                outputs = torch.cat(outputs)
                outputs = outputs[torch.argsort(indices)]
                return outputs

        return wrapper

    def predict(self, probs: Tensor, data: Data, mask: Tensor, mask_name: str):
        data = data.to(self.device)
        with torch.no_grad():
            self.model.metrics.calculate_metrics(probs[mask], data.y[mask], mask_name)
