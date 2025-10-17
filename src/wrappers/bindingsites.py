from typing import Dict, List

import torch
import torchmetrics
from einops._torch_specific import allow_ops_in_compiled_graph
from hydra.utils import instantiate
from torch import Tensor, nn
from torch_geometric.nn.pool import knn
from torchmetrics import AUROC, Accuracy, JaccardIndex

from src.modules.losses import ConfidenceLoss, DiceLoss
from src.modules.metrics import DCA, DCC
from src.utils.misc import calc_group_var, multi_predictions
from src.wrappers.base import WrapperBase

allow_ops_in_compiled_graph()


class BindingSitesLoss(nn.Module):
    def __init__(
        self,
        segmentation_loss: nn.Module = DiceLoss(),
        global_node_pos_loss: nn.Module = nn.HuberLoss(),
        confidence_loss: nn.Module = ConfidenceLoss(),
        segmentation_loss_weight: float = 1.0,
        global_node_pos_loss_weight: float = 1.0,
        confidence_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.segmentation_loss = segmentation_loss
        self.global_node_pos_loss = global_node_pos_loss
        self.confidence_loss = confidence_loss
        self.segmentation_loss_weight = segmentation_loss_weight
        self.global_node_pos_loss_weight = global_node_pos_loss_weight
        self.confidence_loss_weight = confidence_loss_weight

    def forward(self, model, batch):
        pred_seg, pred_pos_global_node, _, preds_confidence = model(batch)

        seg_loss = self.segmentation_loss(pred_seg.squeeze(), batch["atom"].y)

        x = pred_pos_global_node
        y = batch["atom"].bindingsite_center
        x_batch = batch["global_node"].batch
        y_batch = batch["atom"]["bindingsite_center_batch"]

        assign_index = knn(
            x=x,
            y=y,
            batch_x=x_batch,
            batch_y=y_batch,
            k=1,
        )

        dists = torch.norm(y[assign_index[0]] - x[assign_index[1]], dim=-1)
        global_node_pos_loss = self.global_node_pos_loss(
            y[assign_index[0]], x[assign_index[1]]
        )
        pos_var = calc_group_var(x, x_batch).mean()
        conf_var = calc_group_var(preds_confidence, x_batch).mean()

        confidence_assign_index = knn(
            x=y,
            y=x,
            batch_x=y_batch,
            batch_y=x_batch,
            k=1,
        )
        dists_confidence = torch.norm(
            y[confidence_assign_index[1]] - x[confidence_assign_index[0]], dim=-1
        ).detach()

        confidence_loss = self.confidence_loss(
            dists_confidence,
            preds_confidence.squeeze(),
        )

        loss_dict = {
            "dist": dists.mean(),
            "pos_loss": global_node_pos_loss,
            "pos_var": pos_var,
            "seg_loss": seg_loss,
            "confidence_loss": confidence_loss,
            "confidence_var": conf_var,
            "loss": global_node_pos_loss * self.global_node_pos_loss_weight
            + seg_loss * self.segmentation_loss_weight
            + confidence_loss * self.confidence_loss_weight,
        }

        return loss_dict, (pred_seg, pred_pos_global_node, preds_confidence)


class BindingSitesWrapper(WrapperBase):
    """Wrapper that coordinates model, sampling, and training."""

    backbone: nn.Module

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = instantiate(self.hparams.backbone)
        self.backbone = torch.compile(
            self.backbone,
            disable=not self.hparams.compile,
            fullgraph=True,
            dynamic=True,
        )

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

        threshold = self.hparams.threshold
        self.val_dcc = DCC(threshold=threshold)
        self.val_dca = DCA(threshold=threshold)
        self.val_dcc_ranked = DCC(threshold=threshold)
        self.val_dca_ranked = DCA(threshold=threshold)
        self.val_dcc_rand_ranked = DCC(threshold=threshold)
        self.val_dca_rand_ranked = DCA(threshold=threshold)

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        x_atom = batch.x_dict["atom"]
        pos_atom = batch.pos_dict["atom"] / self.hparams.scaling_factor
        x_global_node = batch.x_dict["global_node"]
        pos_global_node = batch.pos_dict["global_node"] / self.hparams.scaling_factor
        edge_index_atom_atom = batch.edge_index_dict[("atom", "to", "atom")]
        edge_index_atom_global_node = batch.edge_index_dict[
            ("atom", "to", "global_node")
        ]
        edge_index_global_node_atom = batch.edge_index_dict[
            ("global_node", "to", "atom")
        ]

        x_atom, pos_global_node, x_global_node, confidence_out = self.backbone(
            x_atom,
            pos_atom,
            x_global_node,
            pos_global_node,
            edge_index_atom_atom,
            edge_index_atom_global_node,
            edge_index_global_node_atom,
        )

        pos_global_node = pos_global_node * self.hparams.scaling_factor
        return x_atom, pos_global_node, x_global_node, confidence_out

    def model_step(self, batch: Dict[str, Tensor]) -> tuple[Dict[str, Tensor], Tensor]:
        loss_dict, preds = self.loss(model=self, batch=batch)
        return loss_dict, preds

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _ = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["global_node"].batch.unique().numel(),
        )
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, preds = self.model_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["global_node"].batch.unique().numel(),
        )

        pred_seg, pred_pos_global_node, preds_confidence = preds

        self.val_seg_metrics(pred_seg.squeeze(), batch["atom"].y)
        self.log_dict(self.val_seg_metrics, on_step=False, on_epoch=True)

        preds_pos_global_node = pred_pos_global_node
        batch_global_nodes = batch["global_node"].batch

        binding_site_center = batch["atom"].bindingsite_center
        self.val_dcc(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
        )
        self.val_dca(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
        )

        self.val_dcc_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
            global_node_confidence=preds_confidence,
        )

        self.val_dca_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
            global_node_confidence=preds_confidence,
        )
        rand_confs = torch.rand_like(preds_confidence)
        self.val_dcc_rand_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_bindingsites=binding_site_center,
            batch_global_nodes=batch_global_nodes,
            batch_bindingsites=batch["atom"]["bindingsite_center_batch"],
            global_node_confidence=rand_confs,
        )
        self.val_dca_rand_ranked(
            coords_global_nodes=preds_pos_global_node,
            coords_ligands=batch["ligand"].ligand_coords,
            ligand_ids=batch["ligand"].ligand_ids,
            batch_global_nodes=batch_global_nodes,
            batch_ligands=batch["ligand"].ligand_coords_batch,
            global_node_confidence=rand_confs,
        )

        self.log(
            "val/dcc",
            self.val_dcc,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca",
            self.val_dca,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dcc_ranked",
            self.val_dcc_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca_ranked",
            self.val_dca_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dcc_rand_ranked",
            self.val_dcc_rand_ranked,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/dca_rand_ranked",
            self.val_dca_rand_ranked,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> List[Dict[str, Tensor]]:
        (
            pred_pos,
            pred_conf,
            batch_global_nodes,
            init_pos,
        ) = multi_predictions(
            model=self,
            batch=batch,
            num_cycles=self.hparams.pred_cycles,
        )

        coords_global_nodes = pred_pos
        protein_names = batch["protein_name"]

        batch_outputs: List[Dict[str, Tensor]] = []
        for batch_global_node, i in enumerate(batch_global_nodes.unique()):
            s_coords = coords_global_nodes[batch_global_node == batch_global_nodes]
            s_confs = pred_conf[batch_global_node == batch_global_nodes]

            output_dict = {
                "protein_name": [protein_names[i]] * s_coords.shape[0],
                "coords": s_coords,
                "confidence": s_confs,
            }

            if self.hparams.save_vn_initial_pos:
                output_dict["vn_initial_pos"] = init_pos[
                    batch_global_node == batch_global_nodes
                ]
            batch_outputs.append(output_dict)

        return batch_outputs
