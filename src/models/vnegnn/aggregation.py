from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation, SoftmaxAggregation
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax


class VariancePreservingAggregation(Aggregation):
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -1,
    ) -> Tensor:
        sorted_indices, argsort = torch.sort(index)
        sorted_x = x[argsort]

        sum_aggregation = self.reduce(
            sorted_x, sorted_indices, ptr, dim_size, dim, reduce="sum"
        )
        counts = self.reduce(
            torch.ones_like(x), sorted_indices, ptr, dim_size, dim, reduce="sum"
        )

        # The replacement with 0 is needed, for the case that a node has no neighbours.
        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))


class HopfieldAggregation(SoftmaxAggregation):
    def __init__(
        self,
        t: float = 1,
        learn: bool = False,
        semi_grad: bool = False,
        feature_dim: int = 50,
    ):
        super().__init__(t, learn, semi_grad)
        # TODO: wanted to use Linear(-1, 1) here, because then we don't need to
        self.att_linear = Linear(feature_dim, 1)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        t = self.t
        att_linear = self.att_linear(x).squeeze()

        alpha = att_linear
        if not isinstance(t, (int, float)) or t != 1:
            alpha = att_linear * t

        if not self.learn and self.semi_grad:
            with torch.no_grad():
                alpha = softmax(att_linear, index, ptr)
        else:
            alpha = softmax(att_linear, index, ptr)
        alpha = alpha.unsqueeze(1)
        return self.reduce(x * alpha, index, ptr, dim_size, dim, reduce="sum")
