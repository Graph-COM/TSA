import functools
import logging
import os
import sys
# import copy
# from torch_geometric.utils import degree

import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
import pdb
import time

from src.adaptation import adapter_manager
from src.data import dataset_manager
from src.model import model_manager
from src.utils import Metrics, set_config_seed, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    metrics = Metrics(cfg.general_config.log_dir, cfg.data_config.name)
    for seed in cfg.general_config.seed:
        run_ex(metrics, seed, **cfg)
    metrics.summarize_results()


def run_ex(
    metrics: Metrics,
    seed: int,
    data_config: DictConfig,
    model_config: DictConfig,
    adapter_config: DictConfig,
    **kwargs,
):
    set_seed(seed)
    set_config_seed(seed, data_config, model_config, adapter_config)
    logger = logging.getLogger(__name__)
    logger.info("Config please see .hydra/config.yaml")

    if data_config.name == "MAG" and data_config.synthetic:
        model_config.input_dim = data_config.num_classes
        data_config.input_dim = data_config.num_classes
        model_config.source = "syn_" + model_config.source
    # Source domain
    dataset = dataset_manager(src_tgt="source", data_config=data_config)
    pre_model = model_manager(metrics=metrics, model_config=model_config)
    pre_model.get_pretrain(dataset)
    source_stats = pre_model.get_src_stats(dataset)

    # source_dataset = copy.deepcopy(dataset)
    if adapter_config.calibration:
        pre_model.calibrate(dataset)

    # Target domain
    dataset = dataset_manager(src_tgt="target", data_config=data_config)
    # pdb.set_trace()

    start_time = time.time()
    result = pre_model.predict(dataset, dataset.tgt_test_mask, "Target_test")
    end_time = time.time()
    infer_time = (end_time - start_time)
    # import pdb; pdb.set_trace()
    torch.save(result, "target_graph.pred.pt")
    adapter = adapter_manager(pre_model, source_stats, adapter_config=adapter_config)

    start_time = time.time()
    probs = adapter.adapt(dataset)
    end_time = time.time()
    adapt_time = (end_time - start_time)
    adapter.predict(probs, dataset, dataset.tgt_val_mask, "Adapted_val")
    adapter.predict(probs, dataset, dataset.tgt_test_mask, "Adapted_test")
    adapter.model.metrics.save_time("infer_time", infer_time)
    adapter.model.metrics.save_time("adapt_time", adapt_time)
    # adapter.model.metrics.save_time("refine_time", refine_time)
    # adapter.model.metrics.save_time("nbr_align_time", nbr_align_time)
    # adapter.model.metrics.save_time("snr_time", snr_time)
    # target_dataset = copy.deepcopy(dataset)

    # degree_src = degree(source_dataset.edge_index[1], num_nodes=source_dataset.x.size(0))
    # degree_tgt = degree(target_dataset.edge_index[1], num_nodes=target_dataset.x.size(0))
    # deg_src_avg = degree_src.float().mean().item()
    # deg_tgt_avg = degree_tgt.float().mean().item()

    # scale1 = adapter.nbr_scale[0].item()
    # scale2 = adapter.nbr_scale[1].item()
    # scale3 = adapter.nbr_scale[2].item()

    # src_label = source_stats['true_src_label_distr']
    # tgt_label = pre_model.cal_label_distr_true(target_dataset)
    # src_edge = source_stats['true_src_edge_distr']
    # tgt_edge = pre_model.cal_edge_distr_true(target_dataset)
    # struct_shift = (torch.sum(torch.abs(src_edge - tgt_edge), dim=1)* tgt_label).sum().item()

    # import numpy as np

    # # Store the values in a dictionary
    # data = {
    #     'deg_src_avg': deg_src_avg,
    #     'deg_tgt_avg': deg_tgt_avg,
    #     'scale1': scale1,
    #     'scale2': scale2,
    #     'scale3': scale3,
    #     'struct_shift': struct_shift,
    # }
    # # Save the dictionary as a numpy file
    # np.save(f'{data_config.name}10.npy', data)
    


if __name__ == "__main__":
    run()
