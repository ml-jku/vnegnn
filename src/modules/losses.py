import torch
from torch import Tensor


def gompertz(x: Tensor, a: float, b: float, c: float) -> Tensor:
    return a * torch.exp(-b * torch.exp(-c * x))


class HuberLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0, scaling_factor: float = 1.0):
        super(HuberLoss, self).__init__()
        self.loss = torch.nn.HuberLoss(delta=delta / scaling_factor)

    def forward(self, pred: Tensor, target: Tensor):
        return self.loss(pred, target)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        """Dice loss.

        Args:
            smooth (int, optional): The smoothing factor for dice loss. Defaults to 1.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, targets):
        probs = torch.sigmoid(input)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_coef = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1 - dice_coef


class ConfidenceLoss(torch.nn.Module):
    def __init__(self, gamma=4, c0=0.001):
        super(ConfidenceLoss, self).__init__()
        self.c0 = c0
        self.gamma = gamma
        self.loss = torch.nn.MSELoss()

    def forward(
        self,
        dists: Tensor,
        confs: Tensor,
    ):
        c = dists.detach().clone()
        c[c <= self.gamma] = 1 - c[c <= self.gamma] / (self.gamma * 2)
        c[c > self.gamma] = self.c0
        return self.loss(confs, c)
