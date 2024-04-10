from collections import Counter
from typing import List

import torch
from einops import rearrange
from torchmetrics import Metric


class DCC(Metric):
    def __init__(self, threshold=4) -> None:
        """DCC metric.

        Args:
            threshold (int, optional): The theshold at which it counts as positive sample.
            Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        # Because it could be the case that non bindingsite center is found if all
        # predictions are below threshold, so we use only the entries where there is no nan in the
        # target, this happens in the calc_predicted_bindingsite_center function in egnn.py
        indices_to_use = ~torch.isnan(preds).any(dim=1)

        self.correct += torch.sum(
            torch.linalg.norm((preds[indices_to_use] - target[indices_to_use]), dim=1)
            < self.threshold
        )
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class DCA(Metric):
    def __init__(self, threshold=4) -> None:
        """DCA metric for a single prediction per batch.
        TODO: Multiple predictions per batch

        Args:
            threshold (int, optional): The theshold at which it counts as positive sample.
            Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, ligand_coords: torch.Tensor, ligand_index: torch.Tensor):
        # Don't sort the indices, because we want to keep the order and pyorch geometric does not guarantee
        # that the order is preserved
        unique_indices, ligand_index_counts = torch.unique(
            ligand_index, sorted=False, return_counts=True
        )
        assert preds.shape[0] == unique_indices.shape[0]

        index = 0
        correct = 0
        for i in range(ligand_index_counts.shape[0]):
            # Get the number of ligand atoms for this protein atom
            coords = ligand_coords[index : index + ligand_index_counts[i]]
            if ~torch.isnan(preds[i]).any():
                correct += int(
                    (torch.linalg.norm((coords - preds[i].unsqueeze(0)), dim=-1) < self.threshold)
                    .any()
                    .item()
                )

            index += ligand_index_counts[i]

        self.correct += correct
        self.total += preds.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class DCCRanked(Metric):
    def __init__(self, threshold=4, rank_descending=True) -> None:
        """DCC metric.

        Args:
            threshold (int, optional): The theshold at which it counts as positive sample.
            Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.rank_descending = rank_descending
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        ranking: torch.tensor,
        names: List[str],
        site_count: Counter,
    ):
        for pos_sample, rank_sample, true_site_sample, n in zip(preds, ranking, target, names):
            num_sites = site_count[n.split("_")[0]]
            allowed_sites = pos_sample[
                torch.argsort(rank_sample, descending=self.rank_descending)[:num_sites]
            ]
            min_dist = torch.min(torch.norm(true_site_sample - allowed_sites, dim=1))
            if min_dist < self.threshold:
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class DCARanked(Metric):
    def __init__(self, threshold=4, rank_descending=True) -> None:
        """DCA metric for a single prediction per batch.
        TODO: Multiple predictions per batch

        Args:
            threshold (int, optional): The theshold at which it counts as positive sample.
            Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.rank_descending = rank_descending
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        ligand_coords: torch.Tensor,
        ligand_index: torch.Tensor,
        ranking: torch.tensor,
        names: List[str],
        site_count: Counter,
    ):
        # Don't sort the indices, because we want to keep the order and pyorch geometric does not guarantee
        # that the order is preserved
        unique_indices, ligand_index_counts = torch.unique(
            ligand_index, sorted=False, return_counts=True
        )
        assert preds.shape[0] == unique_indices.shape[0]

        index = 0
        for i, pos_sample, rank_sample, n in zip(
            range(ligand_index_counts.shape[0]), preds, ranking, names
        ):
            # Get the number of ligand atoms for this protein atom
            coords_sample = ligand_coords[index : index + ligand_index_counts[i]]

            num_sites = site_count[n.split("_")[0]]
            allowed_sites = pos_sample[
                torch.argsort(rank_sample, descending=self.rank_descending)[:num_sites]
            ]

            if torch.any(
                rearrange(
                    torch.norm((coords_sample.unsqueeze(1) - allowed_sites), dim=2), "s c -> (s c)"
                )
                < self.threshold
            ).item():
                self.correct += 1

            self.total += 1
            index += ligand_index_counts[i]

    def compute(self):
        return self.correct.float() / self.total
