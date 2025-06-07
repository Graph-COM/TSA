from omegaconf import DictConfig

from .arxiv import Arxiv
from .csbm import CSBM
from .dblp_acm import DBLP_ACM
from .mag import MAG, Syn_MAG
from .pileup import Pileup


def dataset_manager(src_tgt: str, data_config: DictConfig):
    if src_tgt == "source":
        setting = data_config.source
    elif src_tgt == "target":
        setting = data_config.target

    if data_config.name == "CSBM":
        dataset = CSBM(
            root=data_config.root,
            setting=setting,
            data_config=data_config,
        )
    elif data_config.name == "Pileup":
        signal = setting.split("_")[0]
        pu = setting.split("_")[1]
        dataset = Pileup(
            root=data_config.root, signal=signal, pu=pu, data_config=data_config
        )
    elif data_config.name == "Arxiv":
        dataset = Arxiv(
            root=data_config.root,
            years=setting,
            data_config=data_config,
        )
    elif data_config.name == "DBLP_ACM":
        dataset = DBLP_ACM(
            root=data_config.root,
            name=setting,
            data_config=data_config,
        )
    elif data_config.name == "MAG":
        dataset = MAG(
            root=data_config.root,
            lang=setting,
            data_config=data_config,
        )
    else:
        raise ValueError(f"Dataset {data_config.name} not supported")

    return dataset
