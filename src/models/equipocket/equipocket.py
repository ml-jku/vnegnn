# coding=utf-8
"""
The EquiPocket for our work (EquiPocket: an E(3)-Equivariant Geometric Graph Neural
Network for Ligand Binding Site Prediction)
"""

import torch
from torch import nn
from torch_geometric.nn import MLP, global_max_pool, global_mean_pool, radius_graph

from .baseline_models import Baseline_Models


def get_cutoff_ratio(pos, cutoff, surface_egnn_depth):
    all_ratio = []
    dist = torch.cdist(pos, pos)
    all_atom = pos.shape[0]
    # Add a column of ones as a baseline feature (always 1.0, not all_atom!)
    all_ratio.append(torch.ones(all_atom, 1, dtype=pos.dtype, device=pos.device))
    # Start from i=1 to avoid cutoff*0=0 edge case
    for i in range(1, surface_egnn_depth + 2):
        tmp_result = (dist < cutoff * i).sum(dim=1)
        tmp_result = tmp_result.unsqueeze(dim=-1)
        tmp_result = tmp_result.float() / all_atom
        all_ratio.append(tmp_result)
    all_ratio = torch.concat(all_ratio, dim=1)
    return all_ratio


class EquiPocket(nn.Module):
    def __init__(
        self,
        local_geometric_modeling=True,
        global_structure_modeling="gat_egnn",
        surface_egnn_depth=4,
        cutoff=6,
        dense_attention=True,
        out_depth=2,
        out_features=128,
    ):
        super(EquiPocket, self).__init__()
        self.dense_attention = dense_attention
        self.local_geometric_modeling = local_geometric_modeling
        self.global_structure_modeling = global_structure_modeling
        self.surface_egnn_depth = surface_egnn_depth
        self.dense_attention = dense_attention
        self.cutoff = cutoff
        self.out_depth = out_depth
        self.out_features = out_features
        atom_channels = 16
        bond_channels = 16
        trans_input_features = 0
        # local_geometric_modeling
        if self.local_geometric_modeling == True:
            self.trans_local_geometric_feature = MLP(
                in_channels=7,
                hidden_channels=out_features // 2,
                out_channels=out_features // 2,
                dropout=0.1,
                num_layers=out_depth,
            )
            self.trans_surface_feature = MLP(
                in_channels=14,
                hidden_channels=out_features,
                out_channels=out_features,
                dropout=0.1,
                num_layers=out_depth,
            )
            trans_input_features += 2 * out_features
        # global_structure_modeling
        if self.global_structure_modeling == "gat_egnn":
            self.global_structure_modeling_model = Baseline_Models(
                atom_channels=atom_channels,
                bond_channels=bond_channels,
                out_features=out_features,
                gat_depth=1,
                gcn_depth=0,
                egnn_depth=3,
            )
            trans_input_features += out_features
        # concat feautres
        self.trans_geo_feature = MLP(
            in_channels=trans_input_features,
            hidden_channels=out_features,
            out_channels=out_features,
            dropout=0.1,
            num_layers=out_depth,
        )
        # surface_egnn_depth
        if self.surface_egnn_depth > 0:
            from .surface_egnn import SurfaceEGNN

            self.surface_egnn = SurfaceEGNN(
                in_node_nf=out_features,
                hidden_nf=out_features,
                out_node_nf=out_features,
                n_layers=surface_egnn_depth,
            )
            if self.dense_attention == True:
                self.cal_attention = nn.Sequential()
                attention_in_features = surface_egnn_depth + 2
                mlp = MLP(
                    in_channels=attention_in_features,
                    hidden_channels=out_features,
                    out_channels=surface_egnn_depth + 1,
                    dropout=0.1,
                    num_layers=out_depth,
                )
                self.cal_attention.add_module("cal_attention", mlp)
                self.cal_attention.add_module("sigmoid", nn.Sigmoid())
        # predict
        last_out_feature = out_features * (surface_egnn_depth + 1)
        self.all_out = MLP(
            in_channels=last_out_feature,
            norm=None,
            hidden_channels=last_out_feature,
            out_channels=1,
            dropout=0.1,
            num_layers=out_depth,
        )

    def forward(self, batch_data):
        batch = batch_data.batch
        atom_in_surface = batch_data.atom_in_surface
        pos = batch_data.pos
        surface_center_pos = batch_data.surface_center_pos
        vert_batch = batch_data.vert_batch
        vert_pos = batch_data.vert_pos
        node_embedding = []
        # local geometric embedding
        if self.local_geometric_modeling == True:
            new_pos = batch_data.pos[batch_data.atom_in_surface == 1]
            surface_descriptor = batch_data.surface_descriptor
            local_geometric_embedding = self.trans_local_geometric_feature(
                surface_descriptor
            )
            geometric_embedding = torch.concat(
                [
                    global_mean_pool(local_geometric_embedding, vert_batch),
                    global_max_pool(local_geometric_embedding, vert_batch),
                ],
                dim=-1,
            )
            #
            surface_size = torch.concat(
                [
                    global_mean_pool(surface_descriptor, vert_batch),
                    global_max_pool(surface_descriptor, vert_batch),
                ],
                dim=1,
            )
            surface_size_embedding = self.trans_surface_feature(surface_size)
            node_embedding += [geometric_embedding, surface_size_embedding]

        # global_structure_modeling
        if self.global_structure_modeling != False:
            global_structure_node_embedding_all = self.global_structure_modeling_model(
                batch_data
            )
            global_structure_node_embedding = global_structure_node_embedding_all[
                atom_in_surface == 1
            ]
            node_embedding.append(global_structure_node_embedding)
        node_embedding = torch.concat(node_embedding, dim=1)

        # trans 3 * out_features -> out_features
        node_embedding = self.trans_geo_feature(node_embedding)
        # Surface passing
        if self.surface_egnn_depth > 0:
            new_batch = batch_data.batch[atom_in_surface == 1]
            surface_pos = batch_data.pos[atom_in_surface == 1]
            all_node_embedding = []
            all_node_pos = []
            for i in new_batch.unique():
                tmp_node_embedding = node_embedding[new_batch == i]
                tmp_pos = surface_pos[new_batch == i].clone().detach()
                edge_index = radius_graph(tmp_pos, r=self.cutoff, max_num_neighbors=999)
                edge_attr = torch.ones_like(edge_index[0])
                edge_attr = edge_attr.unsqueeze(dim=1)
                tmp_pos = torch.concat(
                    (
                        surface_pos[new_batch == i].unsqueeze(dim=1),
                        surface_center_pos[new_batch == i].unsqueeze(dim=1),
                    ),
                    dim=1,
                )
                new_node_embedding, new_pos = self.surface_egnn(
                    tmp_node_embedding, tmp_pos, edge_index, edge_index
                )
                all_node_embedding.append(new_node_embedding)
                all_node_pos.append(new_pos)
            # all out
            node_embedding = torch.concat(all_node_embedding, dim=0)
            node_pos = torch.concat(all_node_pos, dim=0)[:, 0]
            if self.surface_egnn_depth > 0 and self.dense_attention == True:
                tmp_cutoff_ratio = get_cutoff_ratio(
                    node_pos, self.cutoff, self.surface_egnn_depth
                )
                # tmp_cutoff_ratio = batch_data.cutoff_ratio[
                #     batch_data.atom_in_surface == 1
                # ]
                cutoff_attention = self.cal_attention(tmp_cutoff_ratio)
                cutoff_attention = torch.repeat_interleave(
                    cutoff_attention, self.out_features, dim=1
                )
                node_embedding = node_embedding * cutoff_attention
        # probability and relative direction
        y_hat = self.all_out(node_embedding)
        angle = None
        if self.surface_egnn_depth > 0:
            angle = node_pos - pos[atom_in_surface == 1]
        return y_hat, angle
