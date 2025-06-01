from omegaconf import DictConfig

from src.utils import Metrics, Registry

MODEL_REGISTRY = Registry("MODEL")


def model_manager(metrics: Metrics, model_config: DictConfig):
    name = model_config.name
    model = MODEL_REGISTRY.get(name)(metrics, model_config)
    return model
