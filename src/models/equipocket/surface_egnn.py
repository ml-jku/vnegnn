#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
The Surface-Egnn for our work (EquiPocket: an E(3)-Equivariant Geometric Graph Neural
Network for Ligand Binding Site Prediction)
"""
import torch
import torch.nn as nn


class MC_E_GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        n_channel,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        normalize=False,
        coords_agg="mean",
        tanh=False,
        attention=False,
        dropout=0.1,
    ):
        super(MC_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.attention = attention

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 6 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)
        if self.coords_agg == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coords_agg parameter" % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord


class SurfaceEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.dense = True

        if self.dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        print(in_edge_nf)
        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                MC_E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    n_channel=2,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=True,
                    dropout=dropout,
                ),
            )

    def forward(
        self,
        h,
        x,
        ctx_edges,
        att_edges,
        ctx_edge_attr=None,
        att_edge_attr=None,
        return_attention=False,
    ):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords, atts = [], [], []
        ctx_states.append(h)
        ctx_coords.append(x)
        for i in range(0, self.n_layers):
            h, x = self._modules[f"gcl_{i}"](h, ctx_edges, x, edge_attr=ctx_edge_attr)
            ctx_states.append(h)
            ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        return h, x


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    diff_atom_pos = coord[row][:, 0, :] - coord[col][:, 0, :]
    distance_atom = diff_atom_pos.norm(dim=1)
    diff_surface_atom_0 = coord[row][:, 1, :] - coord[row][:, 0, :]
    distance_surface_atom_0 = diff_surface_atom_0.norm(dim=1)
    # angle_0
    angel_surface_atom_0 = (diff_surface_atom_0 * diff_atom_pos).sum(dim=1)
    angel_surface_atom_0 = angel_surface_atom_0 / distance_atom
    angel_surface_atom_0 = angel_surface_atom_0 / distance_surface_atom_0
    # angle_1
    diff_surface_atom_1 = coord[col][:, 1, :] - coord[col][:, 0, :]
    distance_surface_atom_1 = diff_surface_atom_1.norm(dim=1)
    angel_surface_atom_1 = (diff_surface_atom_1 * diff_atom_pos).sum(dim=1)
    angel_surface_atom_1 = angel_surface_atom_1 / distance_atom
    angel_surface_atom_1 = angel_surface_atom_1 / distance_surface_atom_1
    # angle_2
    agnle_surface_0_surface_1 = (diff_surface_atom_0 * diff_surface_atom_1).sum(dim=1)
    agnle_surface_0_surface_1 = agnle_surface_0_surface_1 / distance_surface_atom_0
    agnle_surface_0_surface_1 = agnle_surface_0_surface_1 / distance_surface_atom_1
    distance_atom = distance_atom.unsqueeze(-1)
    distance_surface_atom_0 = distance_surface_atom_0.unsqueeze(-1)
    distance_surface_atom_1 = distance_surface_atom_1.unsqueeze(-1)
    angel_surface_atom_0 = angel_surface_atom_0.unsqueeze(-1)
    angel_surface_atom_1 = angel_surface_atom_1.unsqueeze(-1)
    agnle_surface_0_surface_1 = agnle_surface_0_surface_1.unsqueeze(-1)
    radial = torch.concat(
        (
            distance_atom,
            distance_surface_atom_0,
            distance_surface_atom_1,
            angel_surface_atom_0,
            angel_surface_atom_1,
            agnle_surface_0_surface_1,
        ),
        dim=-1,
    )
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    pass
