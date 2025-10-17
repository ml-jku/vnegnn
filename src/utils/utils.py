import importlib
import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger, rich_utils

from .logging_utils import load_run_config_from_wb

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task
    function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception
            (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder
            (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory
            # errors so when using hparam search plugins like Optuna, you might want
            # to disable raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def load_class(class_string):
    module_name, class_name = class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    loaded_class = getattr(module, class_name)
    return loaded_class


def extract_keys_from_nested_dict(conf: dict | DictConfig, keys: list[str]) -> dict:
    found_values = {}
    for key in conf.keys():
        if isinstance(conf[key], dict) or isinstance(conf[key], DictConfig):
            found_values.update(extract_keys_from_nested_dict(conf[key], keys))
        elif key in keys:
            found_values[key] = conf[key]
    return found_values


def replace_values(conf: dict | DictConfig, replacements: dict) -> dict | DictConfig:
    for key in conf.keys():
        if isinstance(conf[key], dict) or isinstance(conf[key], DictConfig):
            conf[key] = replace_values(conf[key], replacements)
        elif key in replacements.keys():
            conf[key] = replacements[key]
    return conf


def load_ckpt_path(ckpt_dir: str, last: bool = False) -> str:
    """Get the checkpoint path based on configuration.

    Args:
        ckpt_dir: Directory containing checkpoint files.
        last: Whether to load the last checkpoint.
    Returns:
        str: Path to the checkpoint file
    """
    if last:
        return os.path.join(ckpt_dir, "last.ckpt")
    else:
        ckpts = [
            f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt") and f != "last.ckpt"
        ]
        if not ckpts:
            raise FileNotFoundError(f"No non-last checkpoints found in {ckpt_dir}")
        if len(ckpts) > 1:
            log.warning(f"Multiple checkpoints found in {ckpt_dir}, using {ckpts[0]}")
        return os.path.join(ckpt_dir, ckpts[0])


def update_cfg_from_first_stage(cfg: DictConfig) -> DictConfig:
    cfg_first_stage = load_run_config_from_wb(
        entity=cfg.first_stage_settings.entity,
        project=cfg.first_stage_settings.project,
        run_id=cfg.first_stage_settings.run_id,
    )
    overwrites = extract_keys_from_nested_dict(
        cfg_first_stage, cfg.first_stage_settings.overwrites
    )
    cfg = replace_values(cfg, overwrites)

    cfg.model.first_stage_model.class_name = cfg_first_stage.model._target_
    if cfg.first_stage_settings.model_path is not None:
        cfg.model.first_stage_model.path = cfg.first_stage.model_path
    else:
        ckpt_dir = cfg_first_stage.callbacks.model_checkpoint.dirpath
        cfg.model.first_stage_model.path = load_ckpt_path(
            ckpt_dir, last=cfg.first_stage_settings.last
        )
    return cfg


def merge_config_section(cfg: DictConfig, cfg_wandb: DictConfig, section: str) -> None:
    """Merge a configuration section from wandb config into the current config.

    :param cfg: Current configuration
    :param cfg_wandb: Configuration loaded from wandb
    :param section: Section name to merge (e.g., 'model', 'data')
    """
    if getattr(cfg, section) is None:
        setattr(cfg, section, getattr(cfg_wandb, section))
    else:
        wandb_section = getattr(cfg_wandb, section)
        current_section = getattr(cfg, section)
        merged_section = OmegaConf.merge(wandb_section, current_section)
        setattr(cfg, section, merged_section)
