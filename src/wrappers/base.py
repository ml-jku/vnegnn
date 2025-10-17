from typing import Any, Dict, Optional, Tuple

import lightning as L
from hydra.utils import instantiate
from torch import Tensor

from src.utils.pylogger import RankedLogger
from src.utils.tensor_utils import tensor_tree_map

log = RankedLogger(__name__, rank_zero_only=True)


class WrapperBase(L.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ema = None
        self.cached_weights = None

    def setup(self, stage: Optional[str] = None):
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        print("Loading EMA weights")
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.state_dict())
        self.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        print("Restoring cached weights")
        if self.cached_weights is not None:
            self.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema:
            self.ema.update(self)

    def on_train_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)

    def on_validation_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_validation_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_test_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_test_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading EMA state dict from checkpoint {checkpoint['epoch']}")
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def configure_optimizers(self) -> Dict[str, Any]:
        self.lr = self.hparams.optimizer.lr
        optimizer = instantiate(self.hparams.optimizer)(
            params=filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        )
        if hasattr(self.hparams, "scheduler") and self.hparams.scheduler is not None:
            scheduler = instantiate(self.hparams.scheduler)(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": self.hparams.interval,
                    "frequency": (
                        self.hparams.frequency
                        if hasattr(self.hparams, "frequency")
                        else 1
                    ),
                },
            }
        return {"optimizer": optimizer}

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return self.backbone(batch, **kwargs)

    def model_step(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        loss, preds = self.loss(model=self, batch=batch)
        return loss, preds

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["target_fields"].size(0),
        )
        return loss
