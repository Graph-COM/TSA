from omegaconf import DictConfig

from src.utils.registry import Registry

ADAPTER_REGISTRY = Registry("ADAPTER")


def adapter_manager(pre_model, source_stats, adapter_config: DictConfig):
    name = adapter_config.name
    if name.startswith("Matcha"):
        name = "Matcha"
    elif name.startswith("TSA"):
        name = "TSA"
    adapter = ADAPTER_REGISTRY.get(name)(pre_model, source_stats, adapter_config)
    return adapter
