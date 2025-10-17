import lightning as pl
import torch
from lightning import Callback


class ConfigLRScheduler(Callback):
    """Count up every gradient update step rather than every epoch."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if len(trainer.lr_scheduler_configs) > 0:
            self.scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert self.scheduler.__class__.__name__ == "LinearWarmupCosineAnnealingLR"
            self.scheduler.set_steps_per_epoch(
                len(trainer.train_dataloader) // trainer.accumulate_grad_batches
            )


class PeakMemory(Callback):
    """Get the maximum memory used during training."""

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset stats
        if "cuda" in str(pl_module.device):
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        # Log the maximum memory consumption
        if "cuda" in str(pl_module.device):
            max_memory_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            self.log("train/max_memory", max_memory_gb, prog_bar=True, sync_dist=True)
