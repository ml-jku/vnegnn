import logging
from typing import List, Tuple

import numpy as np
import torch
from scipy.spatial import distance
from scipy.special import gammainc
from torch import Tensor
from torch.distributions.normal import Normal

log = logging.getLogger(__name__)


def neighbour_nodes(
    coords: np.array, neigh_dist_cutoff: float = 6.5, max_neighbour: int = 10
) -> Tuple[List[int], List[int]]:
    """Compute the neighbour nodes of a given node.

    Args:
        coords (np.array): The coordinates of the nodes.
        neigh_dist_cutoff (float, optional): The maximum distance between two
        nodes to be.
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
        [
            [torch.cos(phi), 0, torch.sin(phi)],
            [0, 1, 0],
            [-torch.sin(phi), 0, torch.cos(phi)],
        ]
    )
    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(psi), -torch.sin(psi)],
            [0, torch.sin(psi), torch.cos(psi)],
        ]
    )
    R = torch.mm(Rz, torch.mm(Ry, Rx))  # Combined rotation matrix
    return R


def sample_fibonacci_grid(
    centroid: Tensor,
    radius: Tensor,
    num_points: int,
    random_rotations: bool = True,
) -> Tensor:
    device = centroid.device
    golden_ratio = (1.0 + torch.sqrt(torch.tensor(5.0, device=device))) / 2.0

    theta = (
        2 * torch.pi * torch.arange(num_points, device=device).float() / golden_ratio
    )
    phi = torch.acos(
        1 - 2 * (torch.arange(num_points, device=device).float() + 0.5) / num_points
    )
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    points = torch.stack((x, y, z), dim=1)
    if random_rotations:
        rotation_matrix = random_rotation_matrix().to(device)
        points = torch.mm(points, rotation_matrix.T)  # Corrected rotation step

    points = centroid + points

    return points


def sample_uniform_in_sphere(centroid: Tensor, radius: Tensor, num_points: int):
    # Use torch.no_grad() to avoid tracking gradients
    r = radius
    ndim = centroid.size(0)
    normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    x = normal_dist.sample((num_points, ndim)).squeeze(-1)
    ssq = torch.sum(x**2, axis=1)
    fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / torch.sqrt(ssq)
    frtiled = fr.unsqueeze(1).repeat(1, ndim)
    p = centroid + x * frtiled

    return p.clone()
