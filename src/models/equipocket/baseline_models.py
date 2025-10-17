"""
Equipocket baseline models
"""

import torch
from torch import nn

# import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv, radius_graph

# from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv, radius_graph

# embed atom embeddings


class embed_atom_chem(torch.nn.Module):
    def __init__(
        self,
        atom_channels=16,
        formal_charge_channels=16,
        chiral_tag_channels=16,
        aromatic_channels=16,
        ring_channels=16,
    ):
        super(embed_atom_chem, self).__init__()
        self.embed_atom_channels = Embedding(100, atom_channels)
        self.embed_charge_channels = Embedding(10, formal_charge_channels)
        self.embed_chiral_tag = Embedding(10, chiral_tag_channels)
        self.embed_aromatic_channels = Embedding(10, aromatic_channels)
        self.embed_ring_channels = Embedding(10, ring_channels)

    def forward(self, x):
        embed_atom_channels_data = self.embed_atom_channels(x[:, 0].long())
        embed_charge_channels_data = self.embed_charge_channels(x[:, 1].long())
        embed_chiral_tag_data = self.embed_chiral_tag(x[:, 2].long())
        embed_aromatic_channels_data = self.embed_aromatic_channels(x[:, 3].long())
        embed_ring_channels_data = self.embed_ring_channels(x[:, 4].long())
        all_atom_embed = torch.cat(
            [
                embed_atom_channels_data,
                embed_charge_channels_data,
                embed_chiral_tag_data,
                embed_aromatic_channels_data,
                embed_ring_channels_data,
                x[:, 5:],
            ],
            axis=1,
        )
        return all_atom_embed


# embedding edge embeddings
class embed_bond_chem(torch.nn.Module):
    def __init__(self, bond_type_channels=16, bond_ring_channels=16):
        super(embed_bond_chem, self).__init__()
        self.embed_bond_type_channels = Embedding(10, bond_type_channels)
        self.embed_bond_ring_channels = Embedding(10, bond_ring_channels)

    def forward(self, edge_attr):
        bond_type_channels_data = self.embed_bond_type_channels(edge_attr[:, 0].long())
        bond_ring_channels_data = self.embed_bond_ring_channels(edge_attr[:, 1].long())
        all_bond_embed = torch.cat(
            [bond_type_channels_data, bond_ring_channels_data, edge_attr[:, 2:]], axis=1
        )
        return all_bond_embed


class Baseline_Models(nn.Module):
    def __init__(
        self,
        atom_channels,
        bond_channels,
        out_features,
        gat_depth=0,
        gcn_depth=0,
        gin_depth=0,
        gcn2_depth=0,
        schnet_depth=0,
        dimenet_depth=0,
        egnn_depth=0,
        cutoff=5,
    ):
        super(Baseline_Models, self).__init__()
        self.gat_depth = gat_depth
        self.gcn_depth = gcn_depth
        self.gin_depth = gin_depth
        self.gcn2_depth = gcn2_depth
        self.schnet_depth = schnet_depth
        self.dimenet_depth = dimenet_depth
        self.egnn_depth = egnn_depth
        # embed atom
        self.get_atom_embed = embed_atom_chem(
            atom_channels=atom_channels,
            formal_charge_channels=atom_channels,
            chiral_tag_channels=atom_channels,
            aromatic_channels=atom_channels,
            ring_channels=atom_channels,
        )
        #  embed edge
        self.get_edge_embed = embed_bond_chem(
            bond_type_channels=bond_channels, bond_ring_channels=bond_channels
        )

        input_atom_length = 5 * atom_channels + 1
        input_edge_length = 2 * bond_channels + 1
        if self.gat_depth > 0:
            self.gat = torch.nn.ModuleList()
            self.gat.append(
                GATConv(
                    in_channels=input_atom_length,
                    out_channels=out_features,
                    heads=1,
                    negative_slope=0.2,
                    dropout=0.1,
                    edge_dim=input_edge_length,
                )
            )
            for i in range(gat_depth - 1):
                self.gat.append(
                    GATConv(
                        in_channels=out_features,
                        out_channels=out_features,
                        heads=1,
                        negative_slope=0.2,
                        dropout=0.1,
                        edge_dim=None,
                    )
                )
        # gat
        if self.gat_depth == 0:
            self.trans = nn.Linear(
                in_features=input_atom_length, out_features=out_features
            )

        # gcn
        if self.gcn_depth > 0:
            self.gcn = torch.nn.ModuleList()
            for i in range(gcn_depth):
                self.gcn.append(
                    GCNConv(in_channels=out_features, out_channels=out_features)
                )

        # gin
        if self.gin_depth > 0:
            self.gin = torch.nn.ModuleList()
            for i in range(gin_depth):
                self.gin.append(
                    GINConv(
                        nn.Linear(in_features=out_features, out_features=out_features),
                        eps=0,
                        train_eps=True,
                    )
                )

        # gcn2
        if self.gcn2_depth > 0:
            self.gcn2 = torch.nn.ModuleList()
            for i in range(gcn2_depth):
                self.gcn2.append(
                    GCN2Conv(channels=out_features, alpha=0.5, shared_weights=False)
                )

        # #  schnet
        # if self.schnet_depth > 0:
        #     from .schnet import SchNet

        #     self.schnet = SchNet(hidden_channels=out_features, cutoff=cutoff)

        # #  dimenet
        # if self.dimenet_depth > 0:
        #     self.dimenet = DimeNet(
        #         hidden_channels=out_features,
        #         out_channels=out_features,
        #         cutoff=cutoff,
        #         num_blocks=6,
        #         num_bilinear=8,
        #         num_spherical=7,
        #         num_radial=6,
        #     )

        # egnn
        if self.egnn_depth > 0:
            from .egnn_clean import EGNN

            self.egnn = EGNN(
                in_node_nf=out_features,
                n_layers=self.egnn_depth,
                hidden_nf=out_features,
                out_node_nf=out_features,
                in_edge_nf=1,
            )

    #
    def forward(self, batch_data):
        new_x = self.get_atom_embed(batch_data.x)
        new_edge_attr = self.get_edge_embed(batch_data.edge_attr)
        # gat
        if self.gat_depth > 0:
            node_embedding = self.gat[0](new_x, batch_data.edge_index, new_edge_attr)
            for tmp_model in self.gat[1:]:
                node_embedding = tmp_model(node_embedding, batch_data.edge_index)
        if self.gat_depth == 0:
            node_embedding = self.trans(new_x)
        # gcn
        if self.gcn_depth > 0:
            for tmp_model in self.gcn:
                node_embedding = tmp_model(node_embedding, batch_data.edge_index)
        # gin
        if self.gin_depth > 0:
            for tmp_model in self.gin:
                node_embedding = tmp_model(node_embedding, batch_data.edge_index)
        # gcn2
        if self.gcn2_depth > 0:
            original_node_embedding = node_embedding
            for tmp_model in self.gcn2:
                node_embedding = tmp_model(
                    x=node_embedding,
                    x_0=original_node_embedding,
                    edge_index=batch_data.edge_index,
                )

        # schnet
        if self.schnet_depth > 0:
            node_embedding = self.schnet(
                z=batch_data.x[:, 0], pos=batch_data.pos, batch=batch_data.batch
            )

        # dimenet
        if self.dimenet_depth > 0:
            node_embedding = self.dimenet(
                z=node_embedding, pos=batch_data.pos, batch=batch_data.batch
            )

        # egnn
        if self.egnn_depth > 0:
            all_node_embedding = []
            for i in batch_data.batch.unique():
                tmp_node_embedding = node_embedding[batch_data.batch == i]
                tmp_pos = batch_data.pos[batch_data.batch == i]
                edge_index = radius_graph(tmp_pos, r=5, max_num_neighbors=8)
                edge_attr = torch.ones_like(edge_index[0])
                edge_attr = edge_attr.unsqueeze(dim=1)
                tmp_node_embedding, tmp_pos = self.egnn(
                    h=tmp_node_embedding,
                    x=tmp_pos,
                    edges=edge_index,
                    edge_attr=edge_attr,
                )
                all_node_embedding.append(tmp_node_embedding)
            node_embedding = torch.concat(all_node_embedding, dim=0)
        return node_embedding
