import itertools
import logging
from abc import ABC
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, HeteroData

from src.utils.constants import ALLOWABLE_RESIDUES
from typing import Union
from src.utils.protein import ProteinInfo

log = logging.getLogger(__name__)


class GraphBuilderBase(ABC):

    def create_graph(self) -> Union[Data, HeteroData]:
        """Create the graph.

        Returns:
            [Data, HeteroData]: The graph.
        """
        try:
            if not self.complex:
                return self.complex_name

            return self._compute_graph()
        except Exception as e:
            # TODO: Change to a more meaningful exception. Value Error seems the problem
            log.warning(f"Error while processing complex {self.complex_name}. Error: {e}")
            return self.complex_name

    def _add_metadata(self, data: Union[Data, HeteroData]):
        raise NotImplementedError("This method should be implemented in derived classes.")


class HeteroGraphBuilder(GraphBuilderBase):

    def __init__(
        self,
        protein_info: ProteinInfo,
        neigh_dist_cutoff: float = 5,
        max_neighbours: int = 10,
        number_of_global_nodes: int = 1,
        esm_features: bool = True,
    ):
        self.protein_info = protein_info
        self.neigh_dist_cutoff = neigh_dist_cutoff
        self.max_neighbours = max_neighbours
        self.number_of_global_nodes = number_of_global_nodes
        self.esm_features = esm_features

    def _compute_graph(self):
        src_list, dst_list = neighbour_nodes(
            self.protein_info.coordiantes,
            neigh_dist_cutoff=self.neigh_dist_cutoff,
            max_neighbour=self.max_neighbours,
        )

        data = HeteroData()
        data["atom"].pos = torch.FloatTensor(np.copy(self.protein_info.coordiantes))

        if self.protein_info.pocket_center is not None:
            data["atom"].y = torch.FloatTensor(np.copy(self.protein_info.res_binding))
            data["atom"].bindingsite_center = torch.FloatTensor(
                np.copy(self.protein_info.pocket_center)
            ).unsqueeze(0)
            data["atom"].bindingsite_center_direction_vector = (
                data["atom"].bindingsite_center - data["atom"].pos
            )

        if self.esm_features:
            data["atom"].x = torch.FloatTensor(np.copy(self.protein_info.res_features))
            data["global_node"].x = torch.FloatTensor(
                np.array(
                    [
                        self.protein_info.res_features.mean(axis=0)
                        for _ in range(self.number_of_global_nodes)
                    ]
                )
            )
        else:
            le = LabelEncoder().fit(ALLOWABLE_RESIDUES)
            data["atom"].x = torch.nn.functional.one_hot(
                torch.tensor(le.transform(self.protein_info.res_names)),
                num_classes=len(le.classes_),
            ).float()
            data["global_node"].x = torch.nn.functional.one_hot(
                torch.tensor(
                    le.transform(["GlobalNode" for _ in range(self.number_of_global_nodes)])
                ),
                num_classes=len(le.classes_),
            ).float()

        # Save center and radius to sample global node positions during training.
        centroid = self.protein_info.coordiantes.mean(axis=0)
        radius = np.max(np.linalg.norm(self.protein_info.coordiantes - centroid, axis=1))
        data.centroid = torch.FloatTensor(centroid)
        data.radius = torch.FloatTensor(np.array(radius)).unsqueeze(0)

        # This is just a placeholder, the actual positions will be sampled during training.
        global_node_starting_positions = sample_global_node_starting_positions(
            centroid=data.centroid, radius=data.radius, num_points=self.number_of_global_nodes
        )

        data["global_node"].pos = torch.FloatTensor(global_node_starting_positions)
        data["atom", "to", "atom"].edge_index = torch.LongTensor([src_list, dst_list])

        # Creates edges between the global node and all the atoms for number_of_global_nodes.
        # Each global_node should be connected to all the atoms. And each atom should be conneted
        # to all the global nodes.
        src_atom = list(
            itertools.chain.from_iterable(
                [list(range(data["atom"].num_nodes)) for i in range(self.number_of_global_nodes)]
            )
        )
        dst_global_node = list(
            itertools.chain.from_iterable(
                [[i] * data["atom"].num_nodes for i in range(self.number_of_global_nodes)]
            )
        )

        if self.protein_info.ligand_coords is not None:
            data["ligand"].ligand_coords = torch.FloatTensor(self.protein_info.ligand_coords)
            data["ligand"].num_nodes = torch.LongTensor([len(self.protein_info.ligand_coords)])

        data["atom", "to", "global_node"].edge_index = torch.LongTensor(
            [src_atom, dst_global_node]
        )
        data["global_node", "to", "atom"].edge_index = torch.LongTensor(
            [dst_global_node, src_atom]
        )

        if self.protein_info.pocket_center is not None:
            assert (
                len(data["atom"].x)
                == len(data["atom"].pos)
                == len(data["atom"].y)
                == len(data["atom"].bindingsite_center_direction_vector)
            )
        else:
            assert len(data["atom"].x) == len(data["atom"].pos)

        data = self._add_metadata(data)
        return data

    def _add_metadata(self, data: Union[Data, HeteroData]):
        """Adds additional data to the graph not dependent on the graph type

        Args:
            data ([Data, HeteroData]): The graph.

        Returns:
            [Data, HeteroData]: The graph with additional metadata.
        """
        data.name = self.protein_info.name
        data.chains = self.protein_info.chains
        data.res_names = self.protein_info.res_names
        data.atom_names = self.protein_info.atom_names
        data.res_numbers = self.protein_info.res_numbers
        return data

    def create_graph(self) -> Union[Data, HeteroData]:
        """Create the graph.

        Returns:
            [Data, HeteroData]: The graph.
        """
        try:
            return self._compute_graph()
        except Exception as e:
            # TODO: Change to a more meaningful exception. Value Error seems the problem
            log.warning(f"Error while processing complex {self.protein_info.name}. Error: {e}")
            return self.protein_info.name


class HomoGraphBuilder(GraphBuilderBase):

    def __init__(
        self,
        protein_info: ProteinInfo,
        neigh_dist_cutoff: float = 5,
        max_neighbours: int = 10,
        number_of_global_nodes: int = 1,
        esm_features: bool = False,
    ):
        self.protein_info = protein_info
        self.neigh_dist_cutoff = neigh_dist_cutoff
        self.max_neighbours = max_neighbours
        self.number_of_global_nodes = number_of_global_nodes
        self.esm_features = esm_features

    def _compute_graph(self):
        src_list, dst_list = neighbour_nodes(
            self.protein_info.coordiantes,
            neigh_dist_cutoff=self.neigh_dist_cutoff,
            max_neighbour=self.max_neighbours,
        )

        data = Data()
        data.x = torch.FloatTensor(np.copy(self.protein_info.res_features))
        data.pos = torch.FloatTensor(np.copy(self.protein_info.coordiantes))
        if self.protein_info.pocket_center is not None:
            data.y = torch.FloatTensor(np.copy(self.protein_info.res_binding))
            data.bindingsite_center = torch.FloatTensor(
                np.copy(self.protein_info.pocket_center)
            ).unsqueeze(0)
            data.bindingsite_center_direction_vector = data.bindingsite_center - data.pos
        data.edge_index = torch.LongTensor([src_list, dst_list])

        if not self.esm_features:
            raise NotImplementedError("Only ESM features are supported at the moment.")

        # Save center and radius to sample global node positions during training.
        centroid = self.protein_info.coordiantes.mean(axis=0)
        radius = np.max(np.linalg.norm(self.protein_info.coordiantes - centroid, axis=1))
        data.centroid = torch.FloatTensor(centroid)
        data.radius = torch.FloatTensor(np.array(radius)).unsqueeze(0)

        # Global Node
        num_atom_nodes = data.x.size(0)
        atom_node_ids = torch.arange(num_atom_nodes, dtype=torch.long)
        global_nodes_edges = []
        for i in range(self.number_of_global_nodes):
            global_node_ids = torch.full((num_atom_nodes,), num_atom_nodes + i, dtype=torch.long)
            global_node_edge_index = torch.stack(
                [
                    torch.cat([atom_node_ids, global_node_ids]),
                    torch.cat([global_node_ids, atom_node_ids]),
                ]
            )
            global_nodes_edges.append(global_node_edge_index)
        global_nodes_edges = torch.cat(global_nodes_edges, dim=1)

        x_global_node = torch.FloatTensor(
            np.array(
                [
                    self.protein_info.res_features.mean(axis=0)
                    for _ in range(self.number_of_global_nodes)
                ]
            )
        )
        data.x = torch.cat([data.x, x_global_node])
        data.pos = torch.cat([data.pos, torch.zeros((self.number_of_global_nodes, 3))])
        data.edge_index = torch.cat([data.edge_index, global_nodes_edges], dim=1)

        data.global_node_register = torch.zeros([data.x.size(0)])
        data.global_node_register[data.x.size(0) - self.number_of_global_nodes :] = 1

        if self.protein_info.ligand_coords is not None:
            data.ligand_coords = torch.zeros((data.x.size(0), 3))  # Initialize with zeros
            data.ligand_coords[: len(self.protein_info.ligand_coords)] = torch.FloatTensor(
                self.protein_info.ligand_coords
            )
            data.ligand_register = torch.zeros([data.x.size(0)])
            data.ligand_register[: len(self.protein_info.ligand_coords)] = 1

        if self.protein_info.pocket_center is not None:
            assert (
                data.num_nodes
                == data.x.size(0)
                == data.pos.size(0)
                == data.global_node_register.size(0)
                == data.ligand_register.size(0)
                == data.y.size(0) + self.number_of_global_nodes
            )
        else:
            assert (
                data.num_nodes
                == data.x.size(0)
                == data.pos.size(0)
                == data.global_node_register.size(0)
            )

        data = self._add_metadata(data)

        return data

    def _add_metadata(self, data: Union[Data, HeteroData]):
        """Adds additional data to the graph not dependent on the graph type

        Args:
            data ([Data, HeteroData]): The graph.

        Returns:
            [Data, HeteroData]: The graph with additional metadata.
        """
        data.name = self.protein_info.name
        data.chains = self.protein_info.chains
        data.res_names = self.protein_info.res_names
        data.atom_names = self.protein_info.atom_names
        data.res_numbers = self.protein_info.res_numbers
        return data

    def create_graph(self) -> Union[Data, HeteroData]:
        """Create the graph.

        Returns:
            [Data, HeteroData]: The graph.
        """
        try:
            return self._compute_graph()
        except Exception as e:
            # TODO: Change to a more meaningful exception. Value Error seems the problem
            log.warning(f"Error while processing complex {self.protein_info.name}. Error: {e}")
            return self.protein_info.name


def neighbour_nodes(
    coords: np.array, neigh_dist_cutoff: float = 6.5, max_neighbour: int = 10
) -> Tuple[List[int], List[int]]:
    """Compute the neighbour nodes of a given node.

    Args:
        coords (np.array): The coordinates of the nodes.
        neigh_dist_cutoff (float, optional): The maximum distance between two nodes to be.
        Defaults to 6.5.
        max_neighbour (int, optional): The maximum number of neighbours. Defaults to 10.

    Returns:
        Tuple[List[int], List[int]]: The source and destination nodes.
        In the format expected by pytorch geometric.
    """
    pairwise_dists = distance.cdist(coords, coords)
    np.fill_diagonal(pairwise_dists, np.inf)

    src_list = []
    dst_list = []

    for i in range(pairwise_dists.shape[0]):
        src = list(np.where(pairwise_dists[i, :] < neigh_dist_cutoff)[0])
        if len(src) > max_neighbour:
            src = list(np.argsort(pairwise_dists[i, :]))[:max_neighbour]
        dst = [i] * len(src)
        src_list.extend(src)
        dst_list.extend(dst)

    return src_list, dst_list


def random_rotation_matrix():
    """Generate a random 3x3 rotation matrix using PyTorch."""
    theta = 2 * torch.pi * torch.rand(1)  # Random rotation around the z-axis
    phi = torch.acos(2 * torch.rand(1) - 1)  # Random rotation around the y-axis
    psi = 2 * torch.pi * torch.rand(1)  # Random rotation around the x-axis

    Rz = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    Ry = torch.tensor(
        [[torch.cos(phi), 0, torch.sin(phi)], [0, 1, 0], [-torch.sin(phi), 0, torch.cos(phi)]]
    )
    Rx = torch.tensor(
        [[1, 0, 0], [0, torch.cos(psi), -torch.sin(psi)], [0, torch.sin(psi), torch.cos(psi)]]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix
    return R


def sample_global_node_starting_positions(
    centroid: torch.tensor,
    radius: torch.tensor,
    num_points: int,
    random_rotations: bool = True,
) -> torch.tensor:
    golden_ratio = (1.0 + torch.sqrt(torch.tensor(5.0))) / 2.0

    theta = 2 * torch.pi * torch.arange(num_points).float() / golden_ratio
    phi = torch.acos(1 - 2 * (torch.arange(num_points).float() + 0.5) / num_points)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    if random_rotations:
        rotation_matrix = random_rotation_matrix()
        points = torch.mm(points, rotation_matrix.T)  # Corrected rotation step

    points = centroid + points

    return points
