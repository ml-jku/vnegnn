from abc import ABC
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric
from einops import rearrange
from torch import Tensor
from torch.optim.adam import Adam as Adam
from torch_geometric.nn import (
    Aggregation,
    MeanAggregation,
    MessagePassing,
    SumAggregation,
)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_geometric.utils import unbatch
from torchmetrics import JaccardIndex

from src.models.utils import CoorsNorm
from src.utils.losses import ConfidenceLoss
from src.utils.metrics import DCA, DCC, DCARanked, DCCRanked


class EGNNLayer(MessagePassing):
    """E(n)-equivariant Message Passing Layer
    Is currently not compatible with the Pytorch Geometric HeteroConv class, because are returning here
    only the updated target nodes features.
    TODO: Change this to conform with general Pytorch Geometric interface.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        out_features: int,
        act: nn.Module,
        dropout: float = 0.5,
        node_aggr: Aggregation = SumAggregation,
        cord_aggr: Aggregation = MeanAggregation,
        residual: bool = True,
        update_coords: bool = True,
        norm_coords: bool = True,
        norm_coors_scale_init: float = 1e-2,
        norm_feats: bool = True,
        initialization_gain: float = 1,
        return_pos: bool = True,
    ):
        super().__init__(aggr=None)
        self.node_aggr = node_aggr()
        self.cord_aggr = cord_aggr()
        self.residual = residual
        self.update_coords = update_coords
        self.act = act
        self.initialization_gain = initialization_gain
        self.return_pos = return_pos

        if (node_features != out_features) and residual:
            raise ValueError(
                "Residual connections are only compatible with the same input and output dimensions."
            )

        self.message_net = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_features),
            nn.Dropout(dropout),
            act(),
            nn.Linear(hidden_features, hidden_features),
        )

        self.update_net = nn.Sequential(
            nn.Linear(node_features + hidden_features, hidden_features),
            nn.Dropout(dropout),
            act(),
            nn.Linear(hidden_features, out_features),
        )

        self.pos_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.Dropout(dropout),
            act(),
            nn.Linear(hidden_features, 1),
        )

        self.node_norm = (
            torch_geometric.nn.norm.LayerNorm(node_features) if norm_feats else nn.Identity()
        )
        self.coors_norm = (
            CoorsNorm(scale_init=norm_coors_scale_init) if norm_coords else nn.Identity()
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            if (type(self.act.func()) is nn.SELU) or (type(self.act) is nn.SELU):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
                nn.init.zeros_(module.bias)
            else:
                # seems to be needed to keep the network from exploding to NaN with greater depths
                nn.init.xavier_normal_(module.weight, gain=self.initialization_gain)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptPairTensor = None,
    ):
        # TODO: Think about a better solution for the residual connection
        if self.residual:
            residual = x if isinstance(x, Tensor) else x[1]
        x_dest, pos = self.propagate(
            edge_index, x=x, pos=pos, edge_attr=edge_attr, edge_weight=edge_weight
        )

        if self.residual:
            x_dest = x_dest + residual

        out = (x_dest, pos) if self.return_pos else x_dest
        return out

    def message(
        self, x_i: Tensor, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, edge_weight: OptTensor = None
    ):
        """Create messages"""
        pos_dir = pos_i - pos_j
        dist = torch.norm(pos_dir, dim=-1, keepdim=True)
        input = [self.node_norm(x_i), self.node_norm(x_j), dist]
        input = torch.cat(input, dim=-1)
        node_message = self.message_net(input)
        pos_message = self.coors_norm(pos_dir) * self.pos_net(node_message)
        if edge_weight is not None:
            node_message = node_message * edge_weight.unsqueeze(-1)
            pos_message = pos_message * edge_weight.unsqueeze(-1)

        return node_message, pos_message

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Tensor = None,
        dim_size: int = None,
    ) -> Tensor:
        node_message, pos_message = inputs
        agg_node_message = self.node_aggr(node_message, index, ptr, dim_size)
        agg_pos_message = self.cord_aggr(pos_message, index, ptr, dim_size)
        return agg_node_message, agg_pos_message

    def update(
        self,
        message: Tuple[Tensor, Tensor],
        x: Union[Tensor, OptPairTensor],
        pos: Union[Tensor, OptPairTensor],
    ):
        node_message, pos_message = message
        x_, pos_ = (x, pos) if isinstance(x, Tensor) else (x[1], pos[1])
        input = torch.cat((x_, node_message), dim=-1)
        x_new = self.update_net(input)
        pos_new = pos_ + pos_message if self.update_coords else pos
        return x_new, pos_new


class EGNN(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        num_layers,
        act,
        dropout=0.5,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        residual=True,
        update_coords=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        initialization_gain=1,
    ):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            layer = EGNNLayer(
                node_features,
                edge_features,
                hidden_features,
                out_features,
                act,
                dropout=dropout,
                node_aggr=node_aggr,
                cord_aggr=cord_aggr,
                residual=residual,
                update_coords=update_coords,
                norm_coords=norm_coords,
                norm_coors_scale_init=norm_coors_scale_init,
                norm_feats=norm_feats,
                initialization_gain=initialization_gain,
            )
            self.layers.append(layer)

    def forward(self, x, pos, edge_index):
        for layer in self.layers:
            x, pos = layer(
                x=x,
                edge_index=edge_index,
                pos=pos,
            )

        return x, pos


class EGNNGlobalNodeHetero(nn.Module):
    """E(n)-equivariant Message Passing Network"""

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        num_layers,
        act,
        dropout=0.5,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        update_coords=True,
        residual=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        initialization_gain=1,
        weight_share=True,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.weight_share = weight_share
        if weight_share:
            # Use a single layer that will be shared across all iterations
            self.shared_layer = self.create_layer(
                node_features,
                edge_features,
                hidden_features,
                out_features,
                act,
                dropout,
                node_aggr,
                cord_aggr,
                update_coords,
                residual,
                norm_coords,
                norm_coors_scale_init,
                norm_feats,
                initialization_gain,
            )
        else:
            # Create a list of layers, one for each iteration
            self.layers = nn.ModuleList(
                [
                    self.create_layer(
                        node_features,
                        edge_features,
                        hidden_features,
                        out_features,
                        act,
                        dropout,
                        node_aggr,
                        cord_aggr,
                        update_coords,
                        residual,
                        norm_coords,
                        norm_coors_scale_init,
                        norm_feats,
                        initialization_gain,
                    )
                    for _ in range(num_layers)
                ]
            )

    def create_layer(
        self,
        node_features,
        edge_features,
        hidden_features,
        out_features,
        act,
        dropout,
        node_aggr,
        cord_aggr,
        update_coords,
        residual,
        norm_coords,
        norm_coors_scale_init,
        norm_feats,
        initialization_gain,
    ):
        # Centralized layer creation logic
        return nn.ModuleDict(
            {
                "atom_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                ),
                "atom_to_global_node": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                ),
                "global_node_to_atom": EGNNLayer(
                    node_features=node_features,
                    edge_features=edge_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    act=act,
                    dropout=dropout,
                    node_aggr=node_aggr,
                    cord_aggr=cord_aggr,
                    residual=residual,
                    update_coords=update_coords,
                    norm_coords=norm_coords,
                    norm_coors_scale_init=norm_coors_scale_init,
                    norm_feats=norm_feats,
                    initialization_gain=initialization_gain,
                ),
            }
        )

    def forward(
        self,
        x_atom,
        pos_atom,
        x_global_node,
        pos_global_node,
        edge_index_atom_atom,
        edge_index_atom_global_node,
        edge_index_global_node_atom,
    ):
        for i in range(self.num_layers):
            layer = self.shared_layer if self.weight_share else self.layers[i]
            x_atom, pos_atom = layer["atom_to_atom"](
                x=(x_atom, x_atom),
                edge_index=edge_index_atom_atom,
                pos=(pos_atom, pos_atom),
            )
            x_global_node, pos_global_node = layer["atom_to_global_node"](
                x=(x_atom, x_global_node),
                edge_index=edge_index_atom_global_node,
                pos=(pos_atom, pos_global_node),
            )
            x_atom, pos_atom = layer["global_node_to_atom"](
                x=(x_global_node, x_atom),
                edge_index=edge_index_global_node_atom,
                pos=(pos_global_node, pos_atom),
            )

        return x_atom, x_global_node, pos_atom, pos_global_node


class EGNNClassifierBase(ABC, pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler=None,
        batch_size: int = None,
        segmentation_loss=nn.BCEWithLogitsLoss(),
        segmentation_loss_weight: int = 0,
        dcc_threshold: int = 4,
    ):
        """Base class for the EGNN classifier. Implements the training loop and the metrics.

        Args:
            model (_type_): The model which is called in the forward method.
            optimizer (_type_, optional): The optimizer used for training.
            Defaults to torch.optim.Adam.
            lr_scheduler (_type_, optional): The learning rate scheduler used for training.
            Defaults to None.
            batch_size (_type_, optional): The batch size used for training. This is neccessary
            because everything gets converted to batched data. Defaults to None.k
            Defaults to nn.BCEWithLogitsLoss().
            segmentation_loss (_type_, optional): The loss function used for the segmentation task.
            segmentation_loss_weight (int, optional): The weight of the segmentation loss.
            Defaults to 0.
            dcc_threshold (int, optional): The threshold used for the DCC metric. Defaults to 4.
        """
        super().__init__()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.segmentation_loss = segmentation_loss
        self.segmentation_loss_weight = segmentation_loss_weight
        self.dcc_threshold = dcc_threshold

        self.model = model
        self.mode_complex_site_counter = {}

    def initialize_metrics(self):
        self.metrics = {
            "train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "val": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "val_train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "test_coach420": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "test_pdbbind2020": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "test_holo4k": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
            "test_holo4k_bioass": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
            },
        }

    def update_metrics(self, mode: str, *args, **kwargs):
        y = kwargs.get("y")
        preds = kwargs.get("preds")
        pos = kwargs.get("pos")
        batch_index = kwargs.get("batch_index")
        loss = kwargs.get("loss")
        segmentation_loss = kwargs.get("segmentation_loss")
        bindingsite_center = kwargs.get("bindingsite_center")

        y = y.int()
        self.log(
            f"{mode}/segmentation_loss",
            segmentation_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/loss",
            loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        for metric_name, metric_obj in self.metrics[mode].items():
            # This check is needed because these metrics have the same interface,
            # other metrics can be called differently
            if metric_name in ["acc", "auroc", "precision", "recall", "iou"]:
                metric_obj.update(preds, y)

        preds_bindingsite_center = self.calc_predicted_bindingsite_center(preds, pos, batch_index)
        self.metrics[mode]["dcc"].update(preds_bindingsite_center, bindingsite_center)

    def compute_and_log_metrics(self, mode: str) -> None:
        for metric_name, metric_obj in self.metrics[mode].items():
            self.log(
                f"{mode}/{metric_name}",
                metric_obj.compute(),
                batch_size=self.batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            metric_obj.reset()

    def forward(self, batch):
        raise NotImplementedError

    def process_step(self, mode, batch, batch_idx, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.process_step("train", batch, batch_idx)
        return loss

    def complex_site_counter(self, mode: str):
        if mode in self.mode_complex_site_counter:
            return self.mode_complex_site_counter[mode]
        else:
            if hasattr(self.trainer.datamodule, "named_val_loaders") and hasattr(
                self.trainer.datamodule, "named_test_loaders"
            ):
                for k, v in self.trainer.datamodule.named_val_loaders.items():
                    if k == mode:
                        self.mode_complex_site_counter[mode] = v.dataset.complex_site_counter
                        return self.mode_complex_site_counter[mode]
                for k, v in self.trainer.datamodule.named_test_loaders.items():
                    if k == mode:
                        self.mode_complex_site_counter[mode] = v.dataset.complex_site_counter
                        return self.mode_complex_site_counter[mode]
            else:
                # This is a hack, because the dataloader is not in the datamodule.
                # Its because the metrics are implemented like this,
                # works fine if only one testdaloder is passed to trainer.test is run.
                dataset = self.trainer.test_dataloaders.dataset
                self.mode_complex_site_counter[mode] = dataset.complex_site_counter
                return self.mode_complex_site_counter[mode]

    def get_val_loader_names(self):
        named_loaders = self.trainer.datamodule.named_val_loaders
        loader_names = list(named_loaders.keys())
        return loader_names

    def get_test_loader_names(self):
        if hasattr(self.trainer.datamodule, "named_test_loaders"):
            named_loaders = self.trainer.datamodule.named_test_loaders
            loader_names = list(named_loaders.keys())
            return loader_names
        # If somebody uses a dataloader not in the datamodule.
        return ["custom"]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loader_names = self.get_val_loader_names()
        current_loader_name = loader_names[dataloader_idx]
        self.process_step(current_loader_name, batch, batch_idx)

    def on_train_epoch_start(self):
        for metric in self.metrics["train"].keys():
            self.metrics["train"][metric].to(self.device)

    def on_train_epoch_end(self) -> None:
        self.compute_and_log_metrics("train")

    def on_validation_epoch_start(self):
        loader_names = self.get_val_loader_names()
        for loader_name in loader_names:
            for metric in self.metrics[loader_name].keys():
                self.metrics[loader_name][metric].to(self.device)

    def on_validation_epoch_end(self) -> None:
        loader_names = self.get_val_loader_names()
        for loader_name in loader_names:
            self.compute_and_log_metrics(loader_name)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loader_names = self.get_test_loader_names()
        current_loader_name = loader_names[dataloader_idx]
        self.process_step(current_loader_name, batch, batch_idx)

    def on_test_epoch_start(self):
        loader_names = self.get_test_loader_names()
        for loader_name in loader_names:
            for metric in self.metrics[loader_name].keys():
                self.metrics[loader_name][metric].to(self.device)

    def on_test_epoch_end(self) -> None:
        loader_names = self.get_test_loader_names()
        for loader_name in loader_names:
            self.compute_and_log_metrics(loader_name)

    def calc_predicted_bindingsite_center(self, preds, pos, batch_index):
        # Scatter is faster than to loop over the samples,
        # but there is a problem if the set of bindinsites is empty

        # TODO: Think about a better solution
        # preds_thresholded = preds >= 0.5
        # preds_center = scatter(
        #    batch["atom"].pos[preds_thresholded],
        #    batch["atom"].batch[preds_thresholded],
        #    reduce="mean",
        #    dim=0,
        # )

        preds_unbatch = unbatch(preds >= 0.5, batch_index)
        pos_unbatch = unbatch(pos, batch_index)
        preds_center = []

        for preds_u, pos_u in zip(preds_unbatch, pos_unbatch):
            preds_center.append(pos_u[preds_u].mean(axis=0))

        return torch.stack(preds_center, dim=0)

    def configure_optimizers(self):
        optimizer_config = {}
        optimizer = self.optimizer(params=self.parameters())
        optimizer_config["optimizer"] = optimizer

        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer=optimizer)
            optimizer_config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "monitor": "val/dcc_global_node_ranked_confidence",
                "frequency": self.trainer.check_val_every_n_epoch,
            }

        return optimizer_config


class EGNNClassifierGlobalNodeHomo(EGNNClassifierBase):
    def __init__(
        self,
        input_features=20,
        node_features=30,
        edge_features=1,
        hidden_features=50,
        out_features=20,
        num_layers=3,
        act=nn.LeakyReLU,
        optimizer=torch.optim.Adam,
        lr_scheduler=None,
        batch_size=None,
        dropout=0.5,
        segmentation_loss=nn.BCEWithLogitsLoss(),
        segmentation_loss_weight=0,
        global_node_pos_loss=nn.MSELoss(),
        global_node_pos_loss_weight=0,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        dcc_threshold=4,
        residual=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        number_of_global_nodes=8,
        initialization_gain=1,
        scaling_factor=5,
        confidence_loss_weight=1,
        confidence_gamma=4,
    ):
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "segmentation_loss",
                "global_node_pos_loss",
            ],
        )

        model = EGNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=num_layers,
            act=act,
            dropout=dropout,
            node_aggr=node_aggr,
            cord_aggr=cord_aggr,
            residual=residual,
            update_coords=True,
            norm_coords=norm_coords,
            norm_coors_scale_init=norm_coors_scale_init,
            norm_feats=norm_feats,
            initialization_gain=initialization_gain,
        )

        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            segmentation_loss_weight=segmentation_loss_weight,
            segmentation_loss=segmentation_loss,
            dcc_threshold=dcc_threshold,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.segmentation_loss = segmentation_loss
        self.segmentation_loss_weight = segmentation_loss_weight
        self.global_node_pos_loss = global_node_pos_loss
        self.global_node_pos_loss_weight = global_node_pos_loss_weight
        self.dcc_threshold = dcc_threshold
        self.input_features = input_features
        self.node_features = node_features
        self.out_features = out_features
        self.act = act
        self.dropout = dropout
        self.number_of_global_nodes = number_of_global_nodes
        self.scaling_factor = scaling_factor
        self.confidence_loss = ConfidenceLoss(gamma=confidence_gamma)
        self.confidence_loss_weight = confidence_loss_weight

        self.initialize_layers()
        self.initialize_metrics()

    def initialize_layers(self):
        self.input_mapping = nn.Linear(self.input_features, self.node_features)
        self.head = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.Dropout(self.dropout),
            self.act(),
            nn.Linear(self.out_features, 1),
        )
        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.Dropout(self.dropout),
            self.act(),
            nn.Linear(self.out_features, 1),
        )

    def initialize_metrics(self):
        # We calculate DCA metric not for training, because it is to slow, so we evaluate
        # only at the same timesteps as valdiation thats why we have a val_train set.

        self.metrics = {
            "train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
            },
            "val": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "val_train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_coach420": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_pdbbind2020": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_holo4k": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_holo4k_single": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
        }

    def update_metrics(
        self,
        mode,
        loss,
        segmentation_loss,
        global_node_pos_loss,
        preds,
        pred_pos_global_node,
        closest_global_node,
        bindingsite_center,
        ligand_coords,
        ligand_coords_index,
        y,
        pos,
        batch_index,
        names,
        confidence_loss,
        ranked_global_node_confidence,
    ):
        y = y.int()
        self.log(
            f"{mode}/segmentation_loss",
            segmentation_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/loss",
            loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/global_node_pos_loss",
            global_node_pos_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/confidence_loss",
            confidence_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        preds_bindingsite_center = self.calc_predicted_bindingsite_center(preds, pos, batch_index)

        self.metrics[mode]["dcc"].update(preds_bindingsite_center, bindingsite_center)
        self.metrics[mode]["dcc_global_node"].update(closest_global_node, bindingsite_center)
        self.metrics[mode]["iou"].update(preds, y)

        # Some metrics are not calculated for training because they are to slow to run every epoch.
        if "dca_global_node" in self.metrics[mode]:
            self.metrics[mode]["dca_global_node"].update(
                closest_global_node, ligand_coords, ligand_coords_index
            )

        complex_counter = self.complex_site_counter(mode)

        if "dcc_global_node_ranked_confidence" in self.metrics[mode]:
            self.metrics[mode]["dcc_global_node_ranked_confidence"].update(
                pred_pos_global_node,
                bindingsite_center,
                ranked_global_node_confidence,
                names,
                complex_counter,
            )

        if "dca_global_node_ranked_confidence" in self.metrics[mode]:
            self.metrics[mode]["dca_global_node_ranked_confidence"].update(
                pred_pos_global_node,
                ligand_coords,
                ligand_coords_index,
                ranked_global_node_confidence,
                names,
                complex_counter,
            )

    def forward(self, batch):
        x = batch.x
        pos = batch.pos
        edge_index = batch.edge_index
        global_node_register = batch.global_node_register

        if self.scaling_factor != 1:
            pos = pos / self.scaling_factor

        x = self.input_mapping(x)

        x, pos = self.model(x, pos, edge_index)
        x_atom = x[global_node_register == 0]
        x_global_node = x[global_node_register == 1]
        pos_global_node = pos[global_node_register == 1]

        x_atom = self.head(x_atom)

        confidence_out = self.confidence_mlp(x_global_node)

        if self.scaling_factor != 1:
            pos_global_node = pos_global_node * self.scaling_factor

        return x_atom, pos_global_node, x_global_node, confidence_out

    def process_step(self, mode, batch, batch_idx, *args, **kwargs):
        # TODO: Restructure that method, its already to big and cluttered.
        x, pred_pos_global_node, _, confidence_out = self.forward(batch)
        pos_global_node = batch.bindingsite_center
        global_node_batch_index = batch.batch[batch.global_node_register == 1]

        (
            global_node_pos_loss,
            _,
        ) = self.calc_global_node_losses(
            pred_pos_global_node=pred_pos_global_node,
            pos_global_node=pos_global_node,
            global_node_batch_index=global_node_batch_index,
        )

        num_global_nodes = int(
            pred_pos_global_node.size(0) / torch.unique(global_node_batch_index).size(0),
        )
        pred_pos_global_node_rearranged = rearrange(
            pred_pos_global_node,
            "(b g) d -> b g d",
            g=num_global_nodes,
        )
        global_node_pocket_dists = torch.norm(
            pos_global_node.unsqueeze(1) - pred_pos_global_node_rearranged, dim=-1
        )
        closest_global_node_indices = torch.argmin(
            global_node_pocket_dists,
            dim=-1,
        )
        closest_global_node = pred_pos_global_node_rearranged[
            torch.arange(pred_pos_global_node_rearranged.size(0)), closest_global_node_indices
        ]

        confidence_loss = self.confidence_loss(
            rearrange(global_node_pocket_dists, "b g -> (b g)"), confidence_out
        )

        segmentation_loss = self.segmentation_loss(x.squeeze(), batch.y)
        loss = (
            self.segmentation_loss_weight * segmentation_loss
            + self.global_node_pos_loss_weight * global_node_pos_loss
            + self.confidence_loss_weight * confidence_loss
        )

        preds = torch.sigmoid(x).squeeze()
        self.update_metrics(
            mode,
            loss=loss,
            segmentation_loss=segmentation_loss,
            global_node_pos_loss=global_node_pos_loss,
            preds=preds,
            pred_pos_global_node=pred_pos_global_node_rearranged,
            closest_global_node=closest_global_node,
            bindingsite_center=batch.bindingsite_center,
            ligand_coords=batch.ligand_coords[batch.ligand_register == 1],
            pos=batch.pos[batch.global_node_register == 0],
            y=batch.y,
            batch_index=batch.batch[batch.global_node_register == 0],
            ligand_coords_index=batch.batch[batch.ligand_register == 1],
            names=batch.name,
            confidence_loss=confidence_loss,
            ranked_global_node_confidence=rearrange(
                confidence_out, "(b g) 1 -> b g", g=num_global_nodes
            ),
        )
        return loss

    def compute_and_log_metrics(self, mode) -> None:
        for metric_name, metric_obj in self.metrics[mode].items():
            self.log(
                f"{mode}/{metric_name}",
                metric_obj.compute(),
                batch_size=self.batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            metric_obj.reset()

    def calc_global_node_losses(
        self,
        pred_pos_global_node,
        pos_global_node,
        global_node_batch_index,
    ):
        global_node_pos_loss, global_node_pos_loss_indices = self.global_node_pos_loss(
            true_positions=pos_global_node,
            pred_positions_global_node=pred_pos_global_node,
            global_node_batch_index=global_node_batch_index,
        )
        return (
            global_node_pos_loss,
            global_node_pos_loss_indices,
        )


class EGNNClassifierGlobalNodeHetero(EGNNClassifierBase):
    def __init__(
        self,
        input_features=20,
        node_features=30,
        edge_features=1,
        hidden_features=50,
        out_features=20,
        num_layers=3,
        act=nn.LeakyReLU,
        optimizer=torch.optim.Adam,
        lr_scheduler=None,
        batch_size=None,
        dropout=0.5,
        segmentation_loss=nn.BCEWithLogitsLoss(),
        segmentation_loss_weight=0,
        global_node_pos_loss=nn.MSELoss(),
        global_node_pos_loss_weight=0,
        node_aggr=SumAggregation,
        cord_aggr=MeanAggregation,
        dcc_threshold=4,
        residual=True,
        norm_coords=True,
        norm_coors_scale_init=1e-2,
        norm_feats=True,
        number_of_global_nodes=8,
        global_node_cat_n_bins=10,
        initialization_gain=1,
        n_attention_ranking_heads=4,
        scaling_factor=5,
        confidence_loss_weight=1,
        confidence_gamma=4,
        weight_share=False,
    ):
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "model",
                "segmentation_loss",
                "global_node_pos_loss",
            ],
        )

        model = EGNNGlobalNodeHetero(
            node_features=node_features,
            edge_features=edge_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=num_layers,
            act=act,
            dropout=dropout,
            node_aggr=node_aggr,
            cord_aggr=cord_aggr,
            residual=residual,
            norm_coords=norm_coords,
            norm_coors_scale_init=norm_coors_scale_init,
            norm_feats=norm_feats,
            initialization_gain=initialization_gain,
            weight_share=weight_share,
        )

        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_size=batch_size,
            segmentation_loss_weight=segmentation_loss_weight,
            segmentation_loss=segmentation_loss,
            dcc_threshold=dcc_threshold,
        )
        self.input_features = input_features
        self.node_features = node_features
        self.out_features = out_features
        self.act = act
        self.dropout = dropout
        self.global_node_pos_loss = global_node_pos_loss
        self.global_node_pos_loss_weight = global_node_pos_loss_weight
        self.number_of_global_nodes = number_of_global_nodes
        self.global_node_cat_n_bins = global_node_cat_n_bins
        self.n_attention_ranking_heads = n_attention_ranking_heads
        self.scaling_factor = scaling_factor
        self.confidence_loss = ConfidenceLoss(gamma=confidence_gamma)
        self.confidence_loss_weight = confidence_loss_weight

        self.initialize_layers()
        self.initialize_metrics()

        if (type(self.act.func()) is nn.SELU) or (type(self.act) is nn.SELU):
            self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear", mode="fan_in")
            nn.init.zeros_(module.bias)

    def initialize_layers(self):
        self.input_mapping = nn.Linear(self.input_features, self.node_features)

        self.head = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.Dropout(self.dropout),
            self.act(),
            nn.Linear(self.out_features, 1),
        )

        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.Dropout(self.dropout),
            self.act(),
            nn.Linear(self.out_features, 1),
        )

    def initialize_metrics(self):
        # We calculate DCA metric not for training, because it is to slow, so we evaluate
        # only at the same timesteps as valdiation thats why we have a val_train set.

        # TODO: This breaks the interface of lightning if you use the test function on a new testset.
        # Use procedure as shown in the readme. Will be resolved in future.
        self.metrics = {
            "train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
            },
            "val": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "val_train": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_coach420": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_pdbbind2020": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "test_holo4k": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
            "custom": {
                "iou": JaccardIndex(task="binary"),
                "dcc": DCC(threshold=self.dcc_threshold),
                "dcc_global_node": DCC(threshold=self.dcc_threshold),
                "dca_global_node": DCA(threshold=self.dcc_threshold),
                "dcc_global_node_ranked_confidence": DCCRanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
                "dca_global_node_ranked_confidence": DCARanked(
                    threshold=self.dcc_threshold, rank_descending=True
                ),
            },
        }

    def update_metrics(
        self,
        mode,
        loss,
        segmentation_loss,
        global_node_pos_loss,
        preds,
        pred_pos_global_node,
        closest_global_node,
        bindingsite_center,
        ligand_coords,
        ligand_coords_index,
        y,
        pos,
        batch_index,
        names,
        confidence_loss,
        ranked_global_node_confidence,
    ):
        y = y.int()
        self.log(
            f"{mode}/segmentation_loss",
            segmentation_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/loss",
            loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/global_node_pos_loss",
            global_node_pos_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{mode}/confidence_loss",
            confidence_loss,
            batch_size=self.batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        preds_bindingsite_center = self.calc_predicted_bindingsite_center(preds, pos, batch_index)

        self.metrics[mode]["dcc"].update(preds_bindingsite_center, bindingsite_center)
        self.metrics[mode]["dcc_global_node"].update(closest_global_node, bindingsite_center)
        self.metrics[mode]["iou"].update(preds, y)

        # Some metrics are not calculated for training because they are to slow to run every epoch.
        if "dca_global_node" in self.metrics[mode]:
            self.metrics[mode]["dca_global_node"].update(
                closest_global_node, ligand_coords, ligand_coords_index
            )

        complex_counter = self.complex_site_counter(mode)

        if "dcc_global_node_ranked_confidence" in self.metrics[mode]:
            self.metrics[mode]["dcc_global_node_ranked_confidence"].update(
                pred_pos_global_node,
                bindingsite_center,
                ranked_global_node_confidence,
                names,
                complex_counter,
            )

        if "dca_global_node_ranked_confidence" in self.metrics[mode]:
            self.metrics[mode]["dca_global_node_ranked_confidence"].update(
                pred_pos_global_node,
                ligand_coords,
                ligand_coords_index,
                ranked_global_node_confidence,
                names,
                complex_counter,
            )

    def forward(self, batch):
        x_atom = batch.x_dict["atom"]
        pos_atom = batch.pos_dict["atom"]
        x_global_node = batch.x_dict["global_node"]
        pos_global_node = batch.pos_dict["global_node"]

        if self.scaling_factor != 1:
            pos_atom = pos_atom / self.scaling_factor
            pos_global_node = pos_global_node / self.scaling_factor

        x_atom = self.input_mapping(x_atom)
        x_global_node = self.input_mapping(x_global_node)

        x_atom, x_global_node, _, pos_global_node = self.model(
            x_atom,
            pos_atom,
            x_global_node,
            pos_global_node,
            batch.edge_index_dict[("atom", "to", "atom")],
            batch.edge_index_dict[("atom", "to", "global_node")],
            batch.edge_index_dict[("global_node", "to", "atom")],
        )

        x_atom = self.head(x_atom)
        confidence_out = self.confidence_mlp(x_global_node)

        if self.scaling_factor != 1:
            pos_global_node = pos_global_node * self.scaling_factor

        return x_atom, pos_global_node, x_global_node, confidence_out

    def process_step(self, mode, batch, batch_idx, *args, **kwargs):
        # TODO: Restructure that method, its already to big and cluttered.
        x, pred_pos_global_node, _, confidence_out = self.forward(batch)
        pos_global_node = batch["atom"].bindingsite_center

        (
            global_node_pos_loss,
            _,
        ) = self.calc_global_node_losses(
            pred_pos_global_node=pred_pos_global_node,
            pos_global_node=pos_global_node,
            global_node_batch_index=batch["global_node"].batch,
        )

        num_global_nodes = int(
            pred_pos_global_node.size(0) / torch.unique(batch["global_node"].batch).size(0),
        )
        pred_pos_global_node_rearranged = rearrange(
            pred_pos_global_node,
            "(b g) d -> b g d",
            g=num_global_nodes,
        )
        global_node_pocket_dists = torch.norm(
            pos_global_node.unsqueeze(1) - pred_pos_global_node_rearranged, dim=-1
        )
        closest_global_node_indices = torch.argmin(
            global_node_pocket_dists,
            dim=-1,
        )
        closest_global_node = pred_pos_global_node_rearranged[
            torch.arange(pred_pos_global_node_rearranged.size(0)), closest_global_node_indices
        ]

        confidence_loss = self.confidence_loss(
            rearrange(global_node_pocket_dists, "b g -> (b g)"), confidence_out
        )

        segmentation_loss = self.segmentation_loss(x.squeeze(), batch["atom"].y)
        loss = (
            self.segmentation_loss_weight * segmentation_loss
            + self.global_node_pos_loss_weight * global_node_pos_loss
            + self.confidence_loss_weight * confidence_loss
        )

        preds = torch.sigmoid(x).squeeze()
        self.update_metrics(
            mode,
            loss=loss,
            segmentation_loss=segmentation_loss,
            global_node_pos_loss=global_node_pos_loss,
            preds=preds,
            pred_pos_global_node=pred_pos_global_node_rearranged,
            closest_global_node=closest_global_node,
            bindingsite_center=batch["atom"].bindingsite_center,
            ligand_coords=batch["ligand"].ligand_coords,
            pos=batch["atom"].pos,
            y=batch["atom"].y,
            batch_index=batch["atom"].batch,
            ligand_coords_index=batch["ligand"].batch,
            names=batch.name,
            confidence_loss=confidence_loss,
            ranked_global_node_confidence=rearrange(
                confidence_out, "(b g) 1 -> b g", g=num_global_nodes
            ),
        )
        return loss

    def compute_and_log_metrics(self, mode) -> None:
        for metric_name, metric_obj in self.metrics[mode].items():
            self.log(
                f"{mode}/{metric_name}",
                metric_obj.compute(),
                batch_size=self.batch_size,
                sync_dist=True,
                add_dataloader_idx=False,
            )
            metric_obj.reset()

    def calc_global_node_losses(
        self,
        pred_pos_global_node,
        pos_global_node,
        global_node_batch_index,
    ):
        global_node_pos_loss, global_node_pos_loss_indices = self.global_node_pos_loss(
            true_positions=pos_global_node,
            pred_positions_global_node=pred_pos_global_node,
            global_node_batch_index=global_node_batch_index,
        )

        return (
            global_node_pos_loss,
            global_node_pos_loss_indices,
        )
