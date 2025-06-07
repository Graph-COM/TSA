import functools
import logging
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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

    # Source domain
    dataset = dataset_manager(src_tgt="source", data_config=data_config)
    pre_model = model_manager(metrics=metrics, model_config=model_config)
    pre_model.get_pretrain(dataset)
    source_stats = pre_model.get_src_stats(dataset)

    # Target domain
    dataset = dataset_manager(src_tgt="target", data_config=data_config)

    _ = pre_model.predict(dataset, dataset.tgt_test_mask, "Target_test")
    adapter = adapter_manager(pre_model, source_stats, adapter_config=adapter_config)

    probs = adapter.adapt(dataset)
    adapter.predict(probs, dataset, dataset.tgt_val_mask, "Adapted_val")
    adapter.predict(probs, dataset, dataset.tgt_test_mask, "Adapted_test")


if __name__ == "__main__":
    run()
