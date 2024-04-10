import torch
import torch.nn.functional as F
from einops import rearrange


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        """Dice loss.

        Args:
            smooth (int, optional): The smoothing factor for dice loss. Defaults to 1.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, targets):
        # Ensure the input are in [0, 1]
        probs = torch.sigmoid(input)

        # Flatten the predictions and targets
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate the Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        # Return the loss, which is 1 minus the Dice coefficient
        return 1 - dice_coef


class DirectionLoss(torch.nn.Module):
    def __init__(self):
        """Direction loss. This is 1 - cosine similarity between the input and the target."""
        super(DirectionLoss, self).__init__()

    def forward(self, input, targets):
        return (1 - F.cosine_similarity(input, targets)).mean()


def compute_difference_vectors(global_nodes, atoms):
    return global_nodes.unsqueeze(1) - atoms


class DirectionMinGlobalNodeLoss(torch.nn.Module):
    def forward(
        self,
        atom_positions: torch.tensor,
        pred_pos_global_node: torch.tensor,
        true_direction_vectors: torch.tensor,
        atom_batch_index: torch.tensor,
        global_node_batch_index: torch.tensor,
    ):
        unique = torch.unique(global_node_batch_index)
        min_cos_loss = 0
        min_cos_loss_indices = []
        for i in unique:
            atom_sample_index = torch.where(atom_batch_index == i)[0]
            global_node_sample_index = torch.where(global_node_batch_index == i)[0]
            true_sample_direction_vectors = true_direction_vectors[atom_sample_index]
            pred_direction_vectors = compute_difference_vectors(
                pred_pos_global_node[global_node_sample_index], atom_positions[atom_sample_index]
            )
            cosine_losses = 1 - F.cosine_similarity(
                true_sample_direction_vectors.unsqueeze(0), pred_direction_vectors, dim=-1
            ).mean(1)
            min_cos_loss_index = torch.argmin(cosine_losses)
            min_cos_loss += cosine_losses[min_cos_loss_index]
            min_cos_loss_indices.append(min_cos_loss_index)

        return min_cos_loss / len(unique), torch.stack(min_cos_loss_indices)


class PositionMinGlobalNodeLoss(torch.nn.Module):
    def forward(
        self,
        true_positions: torch.tensor,
        pred_positions_global_node: torch.tensor,
        global_node_batch_index: torch.tensor,
    ):
        number_of_global_nodes = int(
            pred_positions_global_node.shape[0] / torch.unique(global_node_batch_index).shape[0]
        )
        distances_positions = torch.mean(
            (true_positions[global_node_batch_index] - pred_positions_global_node) ** 2,
            axis=-1,
        )
        distances_positions_rearranged = rearrange(
            distances_positions, "(b g) -> b g", g=number_of_global_nodes
        )
        distance_position_min_index = torch.argmin(distances_positions_rearranged, dim=-1)
        distance_position_min = distances_positions_rearranged[
            torch.arange(distances_positions_rearranged.size(0)), distance_position_min_index
        ]
        return distance_position_min.mean(), distance_position_min_index


class ConfidenceLoss(torch.nn.Module):
    def __init__(self, gamma=4, c0=0.001):
        super(ConfidenceLoss, self).__init__()
        self.gamma = gamma
        self.c0 = c0
        self.loss = torch.nn.MSELoss()

    def forward(
        self,
        pocket_dists: torch.tensor,
        confidence_predictions: torch.tensor,
    ):
        c = pocket_dists.detach().clone()
        c[c <= self.gamma] = 1 - c[c <= self.gamma] / (self.gamma * 2)
        c[c > self.gamma] = self.c0
        return self.loss(confidence_predictions.squeeze(), c)


class PositionMinGlobalNodeHuberLoss(torch.nn.Module):
    def forward(
        self,
        true_positions: torch.tensor,
        pred_positions_global_node: torch.tensor,
        global_node_batch_index: torch.tensor,
    ):
        number_of_global_nodes = int(
            pred_positions_global_node.shape[0] / torch.unique(global_node_batch_index).shape[0]
        )
        distances_positions = F.huber_loss(
            pred_positions_global_node, true_positions[global_node_batch_index], reduction="none"
        ).mean(axis=-1)

        distances_positions_rearranged = rearrange(
            distances_positions, "(b g) -> b g", g=number_of_global_nodes
        )
        distance_position_min_index = torch.argmin(distances_positions_rearranged, dim=-1)
        distance_position_min = distances_positions_rearranged[
            torch.arange(distances_positions_rearranged.size(0)), distance_position_min_index
        ]

        return distance_position_min.mean(), distance_position_min_index
