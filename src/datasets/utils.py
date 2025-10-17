import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.pool import radius_graph

from src.utils.constants import RES_IDS
from src.utils.graph import sample_fibonacci_grid


def to_serializable(obj):
    """Convert an object to a serializable dictionary.

    Args:
        obj: Object to convert

    Returns:
        Serializable dictionary
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj


def res_to_one_hot(
    res_names: npt.NDArray | list[str], res_ids: dict[str, int] = RES_IDS
) -> Tensor:
    """Convert residue names to one-hot encoded features.

    Args:
        res_names: List of residue names

    Returns:
        One-hot encoded feature tensor of shape [num_nodes, len(RES_IDS)]
    """
    if isinstance(res_names, np.ndarray):
        res_names = np.array([res_ids[res_name] for res_name in res_names.tolist()])
    return F.one_hot(torch.from_numpy(res_names), num_classes=len(res_ids))


def cat_features(
    feat1: Tensor | npt.NDArray, feat2: Tensor | npt.NDArray
) -> torch.Tensor:
    """Concatenate two features.

    Args:
        feat1: First feature tensor of shape [num_nodes, dim_feat1]
        feat2: Second feature tensor of shape [num_nodes, dim_feat2]

    Returns:
        feat: Concatenated feature tensor of shape [num_nodes, dim_feat1 + dim_feat2]
    """
    if isinstance(feat1, np.ndarray):
        feat1 = torch.from_numpy(feat1)
    if isinstance(feat2, np.ndarray):
        feat2 = torch.from_numpy(feat2)
    return torch.cat([feat1, feat2], dim=-1)


def pad_to_equal_dim(
    feat1: Tensor | npt.NDArray, feat2: Tensor | npt.NDArray
) -> Tensor:
    """Pad two features to have the same dimension.

    Args:
        feat1: First feature tensor of shape [num_nodes, dim_feat1]
        feat2: Second feature tensor of shape [num_nodes, dim_feat2]

    Returns:
        feat1: First feature tensor of shape [num_nodes, max(dim_feat1, dim_feat2)]
        feat2: Second feature tensor of shape [num_nodes, max(dim_feat1, dim_feat2)]

    Example:
        >>> feat1 = torch.randn(10, 3)
        >>> feat2 = torch.randn(10, 4)
        >>> feat1, feat2 = pad_to_equal_dim(feat1, feat2)
        >>> feat1.shape
        torch.Size([10, 4])
        >>> feat2.shape
        torch.Size([10, 4])
    """
    if isinstance(feat1, np.ndarray):
        feat1 = torch.from_numpy(feat1)
    if isinstance(feat2, np.ndarray):
        feat2 = torch.from_numpy(feat2)

    dim_feat1 = feat1.shape[-1]
    dim_feat2 = feat2.shape[-1]

    if dim_feat1 > dim_feat2:
        return feat1, F.pad(feat2, (0, dim_feat1 - dim_feat2))
    elif dim_feat1 < dim_feat2:
        return feat2, F.pad(feat1, (0, dim_feat2 - dim_feat1))
    else:
        return feat1, feat2


def edge_index_reverse_flow(edge_index: Tensor) -> Tensor:
    """Reverse the flow of an edge index.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
    """
    row, col = edge_index[0], edge_index[1]

    return torch.stack([col, row], dim=0)


@dataclass(frozen=True)
class GraphInfo:
    """Graph information.

    Args:
        max_neighbours: Maximum number of neighbours
        number_of_global_nodes: Number of global nodes
        neigh_dist_cutoff: Neighbour distance cutoff
    """

    max_neighbours: int
    number_of_global_nodes: int
    neigh_dist_cutoff: float


def create_hetero_graph(
    protein_name: str,
    coords: npt.NDArray,
    ligand_coords: npt.NDArray,
    ligand_ids: npt.NDArray,
    res_names: npt.NDArray,
    binding_sites: npt.NDArray,
    binding_residues: npt.NDArray,
    graph_info: GraphInfo,
    res_depths: npt.NDArray | None = None,
    esm_features: npt.NDArray | None = None,
) -> HeteroData:
    """Create a hetero graph.

    Args:
        protein_name: Protein name
        coords: Coordinates
        ligand_coords: Ligand coordinates
        ligand_ids: Ligand ids
        res_names: Residue names
        binding_sites: Binding sites
        binding_residues: Binding residues
        graph_info: Graph information
        res_depths: Residue depths
        esm_features: ESM features
    """
    edge_index = radius_graph(
        torch.from_numpy(coords),
        r=graph_info.neigh_dist_cutoff,
        max_num_neighbors=graph_info.max_neighbours,
    )

    data = HeteroData()
    data["atom"].pos = torch.from_numpy(coords)
    data["atom"].x = res_to_one_hot(res_names)

    data["atom"].res_depths = torch.from_numpy(res_depths[..., None])

    if binding_sites is not None:
        data["atom"].y = torch.from_numpy(binding_residues).float()
        data["atom"].bindingsite_center = torch.from_numpy(binding_sites)
        data["ligand"].ligand_coords = torch.from_numpy(ligand_coords)
        data["ligand"].ligand_ids = torch.from_numpy(ligand_ids)

    if esm_features is not None:
        data["atom"].x = cat_features(data["atom"].x, esm_features)
        data["global_node"].x = torch.from_numpy(
            np.array(
                [
                    esm_features.mean(axis=0)
                    for _ in range(graph_info.number_of_global_nodes)
                ]
            )
        )
    else:
        raise NotImplementedError("Only ESM features are supported at the moment.")

    data["atom"].x, data["global_node"].x = pad_to_equal_dim(
        data["atom"].x, data["global_node"].x
    )

    # Save center and radius to sample global node positions during training.
    centroid = coords.mean(axis=0)
    radius = np.max(np.linalg.norm(coords - centroid, axis=1))
    data.centroid = torch.from_numpy(centroid)
    data.radius = torch.from_numpy(np.array(radius))[None]

    # Sample global node positions, actual positions will be randomly
    # sampled during training.
    global_node_starting_positions = sample_fibonacci_grid(
        centroid=data.centroid,
        radius=data.radius,
        num_points=graph_info.number_of_global_nodes,
    )

    data["global_node"].pos = global_node_starting_positions
    data["atom", "to", "atom"].edge_index = edge_index

    src_atom = list(
        itertools.chain.from_iterable(
            [
                list(range(data["atom"].num_nodes))
                for i in range(graph_info.number_of_global_nodes)
            ]
        )
    )
    dst_global_node = list(
        itertools.chain.from_iterable(
            [
                [i] * data["atom"].num_nodes
                for i in range(graph_info.number_of_global_nodes)
            ]
        )
    )

    data["atom", "to", "global_node"].edge_index = torch.LongTensor(
        [src_atom, dst_global_node]
    )
    data["global_node", "to", "atom"].edge_index = torch.LongTensor(
        [dst_global_node, src_atom]
    )

    assert len(data["atom"].x) == len(data["atom"].pos) == len(data["atom"].y)

    if (
        data["atom"].x.isnan().any()
        or data["atom"].pos.isnan().any()
        or data["atom"].y.isnan().any()
        or data["global_node"].x.isnan().any()
        or data["global_node"].pos.isnan().any()
        or data["atom"].res_depths.isnan().any()
    ):
        raise ValueError("Nans in the graph with protein name: %s", protein_name)

    data.protein_name = protein_name
    data.resnames = res_names
    return data


def load_if_exists(path: Path) -> list[str]:
    if path.exists():
        with open(path, "r") as f:
            return f.read().splitlines()
    return []
