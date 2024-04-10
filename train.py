import os

import hydra
import omegaconf
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def run(cfg: DictConfig):
    import pytorch_lightning as pl
    import torch

    import wandb
    from src.utils.lightning import init_lightning_callbacks

    torch.set_float32_matmul_precision(cfg.precision)

    pl.seed_everything(cfg.seed, workers=True)

    print("Working directory : {}".format(os.getcwd()))
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = instantiate(cfg.logger)(config=config)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    logger.watch(model, log="all")
    callbacks = init_lightning_callbacks(cfg)

    trainer: pl.Trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)
    wandb.finish()


def run_no_log(cfg: DictConfig):
    import pytorch_lightning as pl
    import torch

    from src.utils.lightning import init_lightning_callbacks

    torch.set_float32_matmul_precision(cfg.precision)
    pl.seed_everything(cfg.seed, workers=True)

    print("Working directory : {}".format(os.getcwd()))

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    callbacks = init_lightning_callbacks(cfg)

    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)


@hydra.main(config_path="conf", config_name="config_binding_hetero", version_base="1.2")
def run_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    run(cfg)


if __name__ == "__main__":
    run_model()
