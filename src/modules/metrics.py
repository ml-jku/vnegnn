import torch
from torch_geometric.nn.pool import knn
from torchmetrics import Metric


def cal_unique_lig_index(ligand_ids: torch.Tensor, batch_ligand_ids: torch.Tensor):
    change = ligand_ids[1:] != ligand_ids[:-1]
    change = torch.cat(
        [torch.tensor([0], dtype=torch.bool, device=ligand_ids.device), change]
    )
    unique_lig_index = torch.cumsum(change, dim=0) + batch_ligand_ids
    change = unique_lig_index[1:] != unique_lig_index[:-1]
    change = torch.cat(
        [torch.tensor([0], dtype=torch.bool, device=ligand_ids.device), change]
    )
    return torch.cumsum(change, dim=0)


class DCA(Metric):
    def __init__(
        self,
        threshold: float = 4,
        n: int = 0,
    ) -> None:
        """DCA metric for a single prediction per batch.
        TODO: Multiple predictions per batch

        Args:
            threshold (int, optional): The theshold at which it counts as positive
            sample.
            Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.n = n
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        coords_global_nodes: torch.Tensor,
        coords_ligands: torch.Tensor,
        ligand_ids: torch.Tensor,
        batch_global_nodes: torch.Tensor,
        batch_ligands: torch.Tensor,
        global_node_confidence: torch.Tensor = None,
    ):
        unique_lig_index = cal_unique_lig_index(ligand_ids, batch_ligands)

        if global_node_confidence is not None:
            for batch_lig in batch_ligands.unique():
                sample_lig = unique_lig_index[batch_ligands == batch_lig].unique()
                num_ligs = ligand_ids[batch_ligands == batch_lig].unique().shape[0]

                s_coords = coords_global_nodes[batch_lig == batch_global_nodes]
                s_confs = global_node_confidence[batch_lig == batch_global_nodes]

                to_select = min(num_ligs + self.n, s_confs.shape[0])
                top_k = torch.topk(s_confs.squeeze(), k=to_select)[1]
                top_k_coords = s_coords[top_k]
                if top_k_coords.ndim == 1:
                    top_k_coords = top_k_coords.unsqueeze(0)
                s_coords = top_k_coords

                for lig in sample_lig:
                    lig_coords = coords_ligands[unique_lig_index == lig].float()
                    assign_index = knn(x=s_coords.float(), y=lig_coords.float(), k=1)
                    dists = torch.norm(
                        lig_coords[assign_index[0]] - s_coords[assign_index[1]],
                        dim=-1,
                    )
                    self.correct += (dists <= self.threshold).any(dim=-1)
                    self.total += 1

        else:

            for batch_lig in batch_ligands.unique():
                sample_lig = unique_lig_index[batch_ligands == batch_lig].unique()
                sample_global_node_coords = coords_global_nodes[
                    batch_global_nodes == batch_lig
                ]
                for lig in sample_lig:
                    lig_coords = coords_ligands[unique_lig_index == lig]
                    dists = torch.norm(
                        lig_coords[:, None] - sample_global_node_coords, dim=-1
                    )
                    is_below_threshold = (dists <= self.threshold).any()
                    self.correct += is_below_threshold
                    self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class DCC(Metric):
    def __init__(self, threshold: float = 4, n: int = 0) -> None:
        """DCA metric for a single prediction per batch.
        TODO: Multiple predictions per batch

        Args:
            threshold (int, optional): The theshold at which it counts as positive
            sample.Defaults to 4.
        """
        super().__init__()
        self.threshold = threshold
        self.n = n
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        coords_global_nodes: torch.Tensor,
        coords_bindingsites: torch.Tensor,
        batch_global_nodes: torch.Tensor,
        batch_bindingsites: torch.Tensor,
        global_node_confidence: torch.Tensor = None,
    ):
        if global_node_confidence is not None:
            unique, counts = batch_bindingsites.unique(return_counts=True)
            for un, co in zip(unique, counts):
                sample_conf_global_nodes = global_node_confidence[
                    un == batch_global_nodes
                ]
                sample_coords_global_nodes = coords_global_nodes[
                    un == batch_global_nodes
                ]
                sample_coords_binding_sites = coords_bindingsites[
                    un == batch_bindingsites
                ]

                # TODO: Not good if we have 8 nodes and n=10, we are worser than
                # we could be.
                to_select = min(co + self.n, sample_conf_global_nodes.shape[0])

                top_k = torch.topk(sample_conf_global_nodes.squeeze(), k=to_select)[1]
                top_k_coords = sample_coords_global_nodes[top_k]
                if top_k_coords.ndim == 1:
                    top_k_coords = top_k_coords.unsqueeze(0)
                assign_index = knn(
                    x=top_k_coords.float(), y=sample_coords_binding_sites.float(), k=1
                )
                dists = torch.norm(
                    sample_coords_binding_sites[assign_index[0]]
                    - top_k_coords[assign_index[1]],
                    dim=-1,
                )
                self.correct += (dists <= self.threshold).any(dim=-1).sum()
                self.total += co
        else:
            x = coords_global_nodes.float()
            y = coords_bindingsites.float()
            x_batch = batch_global_nodes
            y_batch = batch_bindingsites

            assign_index = knn(
                x=x,
                y=y,
                batch_x=x_batch,
                batch_y=y_batch,
                k=1,
            )
            dists = torch.norm(y[assign_index[0]] - x[assign_index[1]], dim=1)
            correct = (dists <= self.threshold).sum()

            self.correct += correct
            self.total += coords_bindingsites.shape[0]

    def compute(self):
        return self.correct.float() / self.total
