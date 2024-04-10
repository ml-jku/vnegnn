from typing import List

from hydra.utils import instantiate, log
from omegaconf import DictConfig
from pytorch_lightning import Callback


def init_lightning_callbacks(cfg: DictConfig) -> List[Callback]:
    """Initialize callbacks for pytorch lightning ⚡.

    Args:
        cfg (DictConfig): The configuation for the callbacks.

    Returns:
        List[Callback]: The callbacks
    """
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    return callbacks
