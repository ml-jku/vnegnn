import json
import logging
import os
from hashlib import sha256
from pathlib import Path
from typing import Literal

import lightning as pl
import numpy as np
import torch
from joblib import Parallel, cpu_count, delayed
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .utils import load_if_exists, to_serializable

log = logging.getLogger(__name__)

TEST_DATALOADER_INDICES = {
    "coach420": 0,
    "holo4k": 1,
    "pdbbind2020": 2,
    "sc-pdb": 3,
}


class EquipocketDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        protein_names: list[str],
        label="train",
        n_jobs: int = cpu_count() - 1,
        force_reload: bool = False,
    ):
        self.protein_names = protein_names
        self.label = label
        self.n_jobs = n_jobs

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
        protein_ha = json.dumps(to_serializable(self.protein_names), sort_keys=True)
        full_hash = sha256(protein_ha.encode()).hexdigest()[:8]
        name = f"{full_hash}_{self.label}_equipocket.pt"
        return [name]

    def process(self):
        def process_protein(path: Path):
            try:
                binding_info = np.load(path / "binding_atoms.npz")
                graph = torch.load(path / "protein_graph.pt", weights_only=False)
                graph.y = torch.from_numpy(np.array(binding_info["binding_atoms"]))
                graph.protein_name = path.stem
                if (
                    not graph.y.shape[0]
                    == graph["pos"].shape[0]
                    == graph["atom_in_surface"].shape[0]
                ):
                    raise ValueError(
                        f"Binding atoms shape mismatch for {graph.protein_name}"
                    )
                return graph
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


class EquipocketDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root: str,
        train_valid_split: str,
        n_jobs: int = cpu_count() - 1,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 10,
        force_reload: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.train_valid_split = train_valid_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.force_reload = force_reload
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
            EquipocketDataset(
                root=dataset_path,
                label=mode,
                protein_names=complex_names,
                n_jobs=self.n_jobs,
                force_reload=self.force_reload,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle if mode == "train" else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "coach420": 0,
            "holo4k": 1,
            "pdbbind2020": 2,
        }

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return [
            self._create_dataloader("coach420"),
            self._create_dataloader("holo4k"),
            self._create_dataloader("pdbbind2020"),
        ]
