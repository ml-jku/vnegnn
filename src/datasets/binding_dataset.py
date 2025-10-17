import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Literal

import lightning as pl
import numpy as np
import torch
from einops import repeat
from joblib import Parallel, cpu_count, delayed
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.utils.graph import sample_fibonacci_grid, sample_uniform_in_sphere

from .utils import GraphInfo, create_hetero_graph, load_if_exists, to_serializable

log = logging.getLogger(__name__)

TEST_DATALOADER_INDICES = {
    "coach420": 0,
    "holo4k": 1,
    "pdbbind2020": 2,
    "sc-pdb": 3,
}


class BindingDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        protein_names: list[str],
        graph_info: GraphInfo,
        label="train",
        n_jobs: int = cpu_count() - 1,
        random_rotations: bool = False,
        global_node_subsample_size: float = 1.0,
        sampling_strategy: Literal["fibonacci", "uniform"] = "fibonacci",
        sample_radius: bool = False,
        force_reload: bool = False,
    ):
        self.protein_names = protein_names
        self.graph_info = graph_info
        self.label = label
        self.n_jobs = n_jobs

        self.random_rotations = random_rotations
        self.global_node_subsample_size = global_node_subsample_size
        self.sampling_strategy = sampling_strategy
        self.sample_radius = sample_radius

        super().__init__(root, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        raw_file_names = os.listdir(self.raw_dir)
        if len(self.protein_names[0]) != len(raw_file_names[0]):
            return [f for f in raw_file_names if f[:4] in self.protein_names]
        return [f for f in raw_file_names if f in self.protein_names]

    @property
    def processed_file_names(self):
        graph_info_ha = json.dumps(to_serializable(self.graph_info), sort_keys=True)
        protein_ha = json.dumps(to_serializable(self.protein_names), sort_keys=True)

        graph_info_ha = sha256(graph_info_ha.encode()).hexdigest()
        protein_ha = sha256(protein_ha.encode()).hexdigest()
        full_hash = sha256((graph_info_ha + protein_ha).encode()).hexdigest()[:8]

        name = f"{full_hash}_{self.label}.pt"
        return [name]

    def process(self):
        def process_protein(path: Path):
            try:
                binding_info = np.load(path / "binding.npz")
                coords = binding_info["res_coords"]
                ligand_coords = binding_info["ligand_coords"]
                ligand_ids = binding_info["ligand_ids"]
                res_names = binding_info["res_names"]
                binding_residues = binding_info["binding_residues"]
                binding_sites = binding_info["binding_site_centers"]
                esm_features = np.load(path / "embeddings.npz")["residue_embeddings"]
                res_depths = binding_info["res_depths"]

                return create_hetero_graph(
                    protein_name=path.stem,
                    coords=coords,
                    ligand_coords=ligand_coords,
                    ligand_ids=ligand_ids,
                    res_names=res_names,
                    res_depths=res_depths,
                    binding_sites=binding_sites,
                    binding_residues=binding_residues,
                    esm_features=esm_features,
                    graph_info=self.graph_info,
                )
            except Exception as e:
                log.warning(f"Error in {path}: {e}")
                return path.stem

        log.info(
            "Starting parallel protein-to-graph conversion for %d proteins",
            len(self.raw_file_names),
        )
        results = Parallel(n_jobs=self.n_jobs, verbose=1, timeout=None)(
            delayed(process_protein)(Path(f"{self.raw_dir}/{file_name}"))
            for file_name in tqdm(
                self.raw_file_names,
                desc=f"build_graphs[{self.label}]",
            )
        )

        log.info("Finished proteins to graph.")

        not_parsable = [g for g in results if isinstance(g, str)]
        if len(not_parsable) > 0:
            log.info(
                "Write non parsable complexes to file (%d entries)",
                len(not_parsable),
            )
            with open(self.root / f"not_parsable_{self.label}.txt", "w") as f:
                f.write("\n".join(not_parsable))

        data_graphs = [g for g in results if not isinstance(g, str)]
        data, slices = self.collate(data_graphs)
        torch.save((data, slices), self.processed_paths[0])
        log.info("Finished saving data to %s", self.processed_paths[0])

    def get(self, idx: int) -> Data:
        graph = super().get(idx)
        num_points = graph["global_node"].pos.shape[0]

        if self.sampling_strategy == "fibonacci":
            radius = graph.radius
            if self.sample_radius:
                radius = torch.rand(1) * radius
            graph["global_node"].pos = sample_fibonacci_grid(
                graph.centroid,
                radius,
                num_points,
                random_rotations=self.random_rotations,
            )
        elif self.sampling_strategy == "center":
            centers = repeat(graph["atom"].pos.mean(dim=0), "d -> n d", n=num_points)
            graph["global_node"].pos = centers
            graph["global_node"].x = graph["global_node"].x + torch.randn_like(
                graph["global_node"]["x"]
            )
        else:
            graph["global_node"].pos = sample_uniform_in_sphere(
                graph.centroid, graph.radius, num_points
            )

        if self.global_node_subsample_size < 1.0:
            global_node_edge_index = graph[
                "global_node", "to", "atom"
            ].edge_index  # noqa: F821
            new_edge_index = self.random_edge_subset(
                global_node_edge_index, self.global_node_subsample_size
            )
            graph["global_node", "to", "atom"].edge_index = torch.stack(
                new_edge_index
            )  # noqa: F821
            graph["atom", "to", "global_node"].edge_index = torch.stack(
                [new_edge_index[1], new_edge_index[0]]
            )

        graph["atom"].x = graph["atom"].x.float()
        graph["atom"].y = graph["atom"].y.float()
        graph["atom"].pos = graph["atom"].pos.float()
        graph["atom"].bindingsite_center = graph["atom"].bindingsite_center.float()

        graph["global_node"].x = graph["global_node"].x.float()
        graph["global_node"].pos = graph["global_node"].pos.float()

        graph["ligand"].ligand_coords = graph["ligand"].ligand_coords.float()
        graph["ligand"].ligand_ids = graph["ligand"].ligand_ids.float()
        graph["atom"].res_depths = graph["atom"].res_depths.float()

        return graph

    @staticmethod
    def random_edge_subset(edge_index: torch.Tensor, subsample_size: float):
        """Randomly subsample edges from edge_index.

        Args:
            edge_index: Edge index tensor of shape [2, num_edges]
            subsample_size: Fraction of edges to keep (0 to 1)

        Returns:
            Tuple of (src, dst) edge lists
        """
        num_edges = edge_index.shape[1]
        num_keep = int(num_edges * subsample_size)

        # Sample random indices
        perm = torch.randperm(num_edges)[:num_keep]

        # Select edges
        new_edge_index = edge_index[:, perm]

        return new_edge_index[0], new_edge_index[1]


class BindingDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str,
        graph_info: GraphInfo,
        global_node_subsample_size: float = 1.0,
        random_rotations: bool = True,
        sampling_strategy: str = "fibonacci",
        sample_radius: bool = False,
        train_valid_split: float = 0,
        n_jobs: int = cpu_count() - 1,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 10,
        force_reload: bool = False,
        follow_batch: list[str] = [
            "ligand",
            "ligand_coords",
            "bindingsite_center",
            "ligand_ids",
        ],
    ):
        super().__init__()
        self.root = Path(root)
        self.graph_info = graph_info
        self.global_node_subsample_size = global_node_subsample_size
        self.random_rotations = random_rotations
        self.sampling_strategy = sampling_strategy
        self.sample_radius = sample_radius
        self.train_valid_split = train_valid_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.force_reload = force_reload
        self.follow_batch = follow_batch
        self.n_jobs = n_jobs

    def _create_dataloader(
        self, mode: Literal["train", "valid", "coach420", "holo4k"]
    ) -> DataLoader:
        match mode:
            case "train" | "valid":
                dataset_path = self.root / "sc-pdb"
                complex_names_path = (
                    dataset_path / "splits" / f"{mode}_ids_{self.train_valid_split}"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

                blacklist_path = dataset_path / "splits" / "scPDB_blacklist.txt"
                leakage_path = dataset_path / "splits" / "scPDB_leakage.txt"
                blacklist = load_if_exists(blacklist_path)
                leakage = load_if_exists(leakage_path)

                complex_names = [
                    c for c in complex_names if c not in blacklist and c not in leakage
                ]
            case "coach420" | "holo4k" | "pdbbind2020":
                dataset_path = self.root / mode
                complex_names_path = dataset_path / "splits" / f"test_ids_{mode}"
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return DataLoader(
            BindingDataset(
                root=dataset_path,
                label=mode,
                protein_names=complex_names,
                graph_info=self.graph_info,
                global_node_subsample_size=(
                    self.global_node_subsample_size if mode == "train" else 1.0
                ),
                random_rotations=self.random_rotations,
                sampling_strategy=self.sampling_strategy,
                sample_radius=self.sample_radius if mode == "train" else False,
                n_jobs=self.n_jobs,
                force_reload=self.force_reload,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle if mode == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
        )

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
        ]


class BindingEpDataModule(BindingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "pdbbind2020": 2,
        }

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
            self._create_dataloader("pdbbind2020"),
        ]


class BindingPDBTrainDataModule(BindingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_dataloader(
        self, mode: Literal["train", "valid", "coach420", "holo4k"]
    ) -> DataLoader:
        match mode:
            case "train" | "valid":
                dataset_path = self.root / "pdbbind2020"
                complex_names_path = (
                    dataset_path / "splits" / f"{mode}_ids_{self.train_valid_split}"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()

            case "coach420" | "holo4k" | "sc-pdb":
                dataset_path = self.root / mode
                complex_names_path = (
                    dataset_path / "splits" / f"test_ids_{mode}_pdbbind2020"
                )
                with open(complex_names_path, "r") as f:
                    complex_names = f.read().splitlines()
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        return DataLoader(
            BindingDataset(
                root=dataset_path,
                label=mode,
                protein_names=complex_names,
                graph_info=self.graph_info,
                global_node_subsample_size=(
                    self.global_node_subsample_size if mode == "train" else 1.0
                ),
                random_rotations=self.random_rotations,
                sampling_strategy=self.sampling_strategy,
                sample_radius=self.sample_radius if mode == "train" else False,
                n_jobs=self.n_jobs,
                force_reload=self.force_reload,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle if mode == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            follow_batch=self.follow_batch,
        )

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "sc-pdb": 2,
        }

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
            self._create_dataloader("sc-pdb"),
        ]
