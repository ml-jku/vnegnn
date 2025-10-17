from typing import Dict, List

import torch
import torchmetrics
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from hydra.utils import instantiate
from torch import Tensor, nn
from torchmetrics import AUROC, Accuracy, JaccardIndex

from src.modules.losses import DiceLoss
from src.wrappers.base import WrapperBase

allow_ops_in_compiled_graph()


class EquiPocketLoss(nn.Module):
    def __init__(
        self,
        segmentation_loss: nn.Module = DiceLoss(),
    ):
        super().__init__()
        self.segmentation_loss = segmentation_loss

    def forward(self, model, batch):
        y_hat, angle = model(batch)
        y = batch.y[batch["atom_in_surface"] == 1]

        seg_loss = self.segmentation_loss(y_hat, y)

        loss_dict = {
            "seg_loss": seg_loss,
            "loss": seg_loss,
        }

        return loss_dict, (y_hat, angle)


class EquiPocketWrapper(WrapperBase):
    """Wrapper that coordinates model, sampling, and training."""

    backbone: nn.Module

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = instantiate(self.hparams.backbone)
        # self.backbone = torch.compile(
        #     self.backbone,
        #     disable=not self.hparams.compile,
        #     fullgraph=True,
        #     dynamic=True,
        # )

        self.loss = instantiate(self.hparams.loss)
        metrics = torchmetrics.MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "auroc": AUROC(task="binary"),
                "iou": JaccardIndex(task="binary"),
            }
        )
        self.train_seg_metrics = metrics.clone(prefix="train/")
        self.val_seg_metrics = metrics.clone(prefix="val/")

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.backbone(batch)

    def model_step(self, batch: Dict[str, Tensor]) -> tuple[Dict[str, Tensor], Tensor]:
        loss_dict, preds = self.loss(model=self, batch=batch)
        return loss_dict, preds

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, preds = self.model_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=self.hparams.batch_size,
        )

        y_hat, _ = preds
        y = batch.y[batch["atom_in_surface"] == 1]

        self.val_seg_metrics(y_hat.squeeze(), y)
        self.log_dict(self.val_seg_metrics, on_step=False, on_epoch=True)

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> List[Dict[str, Tensor]]:

        y_hat, _ = self(batch)
        protein_names = batch["protein_name"]
        protein_coors_mean = rearrange(batch["mean_pos"], "(b d) -> b d", d=3)

        batch_outputs: List[Dict[str, Tensor]] = []
        for i, batch_id in enumerate(batch.batch.unique()):
            s_confs = y_hat[batch.batch[batch.atom_in_surface == 1] == batch_id]
            s_confs = torch.sigmoid(s_confs)
            s_coords = batch.pos[
                (batch.atom_in_surface == 1) & (batch.batch == batch_id)
            ]
            s_coords = s_coords[s_confs.squeeze() > 0.5]
            s_confs = s_confs[s_confs > 0.5]

            output_dict = {
                "protein_name": [protein_names[i]] * s_coords.shape[0],
                "coords": s_coords + protein_coors_mean[i],
                "confidence_0": s_confs,
            }

            batch_outputs.append(output_dict)

        return batch_outputs
