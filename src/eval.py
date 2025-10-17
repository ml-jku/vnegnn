from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import lightning as pl
import pandas as pd
import rootutils
import torch
from einops._torch_specific import allow_ops_in_compiled_graph
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from tqdm.auto import tqdm

allow_ops_in_compiled_graph()

torch.set_float32_matmul_precision("high")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_loggers,
    load_ckpt_path,
    load_run_config_from_wb,
    log_hyperparameters,
    merge_config_section,
    task_wrapper,
)
from src.utils.misc import (  # noqa: E402
    compute_metric_ratios,
    evaluate_all_proteins,
    predictions_to_df,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    for (
        dataloader_name,
        dataloader_index,
    ) in tqdm(datamodule.test_dataloader_indices.items()):
        log.info(f"Evaluating {dataloader_name}...")
        dataloader = datamodule.test_dataloader()[dataloader_index]
        dataset = dataloader.dataset

        predictions = trainer.predict(
            model=model,
            dataloaders=dataloader,
            ckpt_path=cfg.ckpt_path,
        )
        predictions_df = predictions_to_df(predictions)
        predictions_df.to_csv(
            run_dir / f"predictions_{dataloader_name}.csv", index=False
        )

        res = evaluate_all_proteins(
            df=predictions_df,
            protein_path=Path(dataset.raw_dir),
            num_global_nodes=8,
        )
        df_res = pd.DataFrame(res)
        _, df_rank_dca, dca_ratio_cols = compute_metric_ratios(df_res, "dca")
        _, df_rank_dcc, dcc_ratio_cols = compute_metric_ratios(df_res, "dcc")

        dca_mean = df_rank_dca[dca_ratio_cols].mean(axis=0).to_dict()
        dcc_mean = df_rank_dcc[dcc_ratio_cols].mean(axis=0).to_dict()
        dca_mean = {f"{dataloader_name}/{k}": v for k, v in dca_mean.items()}
        dcc_mean = {f"{dataloader_name}/{k}": v for k, v in dcc_mean.items()}

        logger[0].log_metrics(dca_mean)
        logger[0].log_metrics(dcc_mean)

    metric_dict = trainer.callback_metrics
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    cfg_wandb = load_run_config_from_wb(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        run_id=cfg.wandb_run_id,
    )

    merge_config_section(cfg, cfg_wandb, "model")
    merge_config_section(cfg, cfg_wandb, "data")

    if hasattr(cfg, "ckpt_path"):
        cfg.ckpt_path = load_ckpt_path(
            ckpt_dir=cfg_wandb.callbacks.model_checkpoint.dirpath,
            last=cfg.ckpt_last,
        )

    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
