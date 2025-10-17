from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.data import default_collate


def parse_epoch_outputs(outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Parse the outputs of an epoch. The outputs are expected to be a list of
    dictionaries, where each dictionary contains the outputs of one step. The outputs
    are expected to be in the format of the default_collate function.

    :param outputs: The outputs of the epoch.
    :type outputs: List[Dict[str, Tensor]]
    :return: The parsed outputs. The tensors are rearranged to have the
    batch dimension first.
    :rtype: Dict[str, Tensor]
    """
    return {
        k: rearrange(v, "b h ... -> (b h) ...")
        for k, v in default_collate(outputs).items()
    }


def select_single_sample_index(
    idx: int, step_outputs: List[Dict[str, Tensor]]
) -> Tuple[int, int]:
    """This is for selecting a single sample from the step outputs. Because
    torch_geometric DataLoader batches the data, we need to find the correct batch and
    index within the batch for the given global index.

    :param idx: The global index in the dataset
    :type idx: int
    :param step_outputs: The outputs from one step of the model. Has to contain the key
        "batch_size" for each batch.
    :type step_outputs: List[Dict[str, Tensor]]
    :raises ValueError: _description_
    :return: The batch index and the local index within the batch.
    :rtype: Tuple[int, int]
    """

    global_idx = 0
    for batch_idx, batch_data in enumerate(step_outputs):
        batch_size = batch_data["batch_size"]
        if global_idx + batch_size > idx:
            local_idx = idx - global_idx
            return batch_idx, local_idx
        global_idx += batch_size
    else:
        raise ValueError("Index is out of range.")


def sample_gaussians(points: Tensor, std: float, num_of_samples: int) -> Tensor:
    """Sample points from a Gaussian distribution around the given points.

    :param points: The points to sample around.
    :type points: torch.Tensor
    :param std: The standard deviation of the Gaussian distribution.
    :type std: float
    :param num_of_samples: The number of points to sample around each point.
    :type num_of_samples: int
    :return: The sampled points and the distances to the original points.
    :rtype: torch.Tensor
    """
    num_points = points.shape[0]
    samples = points.repeat_interleave(num_of_samples, dim=0) + std * torch.randn(
        num_points * num_of_samples, points.shape[1]
    )

    return samples


def batch_to_device(
    batch: Dict[str, Tensor], device: str, drop_keys: List[str] = []
) -> Dict[str, Tensor]:
    for key in batch.keys():
        if key not in drop_keys and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = batch_to_device(batch[key], device, drop_keys)
    return batch


def expand_like(a: Tensor, b: Tensor) -> Tensor:
    """Function to reshape a to broadcastable dimension of b.

    Args:
      a (Tensor): [batch_dim, ...]
      b (Tensor): [batch_dim, ...]

    Returns:
      Tensor: Reshaped tensor.
    """
    dims = [1] * (len(b.size()) - len(a.size()))
    a = a.view(*a.shape, *dims)
    return a


def replace_with_mask(a: Tensor, b: Tensor, mask: Tensor) -> Tensor:
    """Replace a with b where mask is True."""
    return torch.where(expand_like(mask, a), a, b)


def context_from_mask(x: Tensor, context_mask: Tensor) -> Tensor:
    """Create new context from the mask and the condition."""
    return replace_with_mask(x, torch.zeros_like(x), context_mask)


@torch.no_grad()
def zero_bias(model: nn.Module):
    for name, p in model.named_parameters():
        if "bias" in name and isinstance(p, nn.Parameter):
            p.zero_()
            p.requires_grad = False


def to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array.

    Handles detaching from computation graph and moving to CPU.
    """
    return tensor.detach().cpu().numpy()
