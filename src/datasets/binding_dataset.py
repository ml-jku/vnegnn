import logging
import os
import pickle
from collections import Counter

import lmdb
import pytorch_lightning as pl
import torch
from joblib import cpu_count
from torch_geometric.data import Data, InMemoryDataset

from src.utils.common import (
    parallel_class_method_execution,
    partial_hash,
    read_strings_from_txt,
)
from src.utils.graph import GraphBuilderBase, sample_global_node_starting_positions

log = logging.getLogger(__name__)


class BindingDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        complex_names_path: str,
        graph_builder: GraphBuilderBase,
        label="train",
        n_jobs: int = cpu_count() - 1,
        random_rotations: bool = False,
        global_node_subsample_size: float = 1.0,
        debug: bool = False,
    ):
        self.complex_names_path = complex_names_path
        self.graph_builder = graph_builder
        self.label = label
        self.n_jobs = n_jobs
        self.debug = debug
        if self.debug:
            self.complex_names_path = f"{self.complex_names_path}_debug"
            self.n_jobs = 1
        self.complex_names = read_strings_from_txt(self.complex_names_path)
        self.complex_site_counter = Counter(
            [complex_name.split("_")[0] for complex_name in self.complex_names]
        )
        self.random_rotations = random_rotations
        self.global_node_subsample_size = global_node_subsample_size

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        hash = partial_hash(self.graph_builder)
        name = f"{self.graph_builder.func.__name__}_{hash.hexdigest()}_{self.label}"
        if self.debug:
            name += "_debug"
        name += ".pt"
        return [name]

    def process(self):
        log.info(
            f"Processing [{self.label}] complexes from [{self.complex_names_path}] "
            f"and saving it to [{(self.complex_names_path)}]"
        )
        proteins = []
        save_path = os.path.join(self.root, "raw", "protein_info.lmdb")
        env = lmdb.open(save_path, create=False)
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                name = key.decode()
                if name in self.complex_names:
                    protein = pickle.loads(value)
                    proteins.append(protein)

        data = parallel_class_method_execution(
            ClassToInitialize=self.graph_builder,
            method_name="create_graph",
            variable_args=proteins,
            n_jobs=self.n_jobs,
            desc="Create protein graphs",
            constant_args=self.graph_builder.keywords,
        )
        log.info("Finished proteins to graph.")

        not_parsable = [g for g in data if isinstance(g, str)]

        log.info("Write non parsable complexes to file")
        log.warn(f"{len(not_parsable)} complexes couldn't be parsed.")
        with open(os.path.join(self.root, f"not_parsable_{self.label}.txt"), "w") as f:
            f.write("\n".join(not_parsable))

        data = [g for g in data if not isinstance(g, str)]

        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])
        log.info("Finished saving data.")

    def get(self, idx: int) -> Data:
        graph = super().get(idx)
        num_points = graph["global_node"].pos.shape[0]

        graph["global_node"].pos = sample_global_node_starting_positions(
            graph.centroid, graph.radius, num_points, random_rotations=self.random_rotations
        )

        if self.global_node_subsample_size < 1.0:
            global_node_edge_index = graph["global_node", "to", "atom"].edge_index
            new_edge_index = self.random_edge_subset(
                global_node_edge_index, self.global_node_subsample_size
            )
            graph["global_node", "to", "atom"].edge_index = torch.stack(new_edge_index)
            graph["atom", "to", "global_node"].edge_index = torch.stack(
                [new_edge_index[1], new_edge_index[0]]
            )

        return graph


class BindingDatasetHomo(BindingDataset):
    def get(self, idx: int) -> Data:
        graph = InMemoryDataset.get(self, idx)
        global_node_register = graph.global_node_register == 1
        num_points = torch.sum(global_node_register)

        graph.pos[global_node_register] = sample_global_node_starting_positions(
            graph.centroid, graph.radius, num_points, random_rotations=self.random_rotations
        )

        if self.global_node_subsample_size < 1.0:
            raise ValueError("For this class the global_node_subsample_size is not implemented.")

        return graph


class PDBBindStaerkDataModuleHetero(pl.LightningDataModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader):
        super().__init__()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader

        # Some metrics should not be evaluated every epoch, because they are very slow.
        # Therefore we have two val dataloaders, where one is the same as the train dataloader.
        # Thats why we get a warning for validation dataloder that we should switch off shuffling.
        self.named_val_loaders = {
            "val": val_dataloader,
            "val_train": train_dataloader,
        }
        self.named_test_loaders = {
            "test": test_dataloader,
        }

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return list(self.named_val_loaders.values())

    def test_dataloader(self):
        return list(self.named_test_loaders.values())


class BindingDataModuleHetero(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader_pdbbind2020,
        test_dataloader_coach420,
        test_dataloader_holo4k,
    ):
        super().__init__()
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader

        # Some metrics should not be evaluated every epoch, because they are very slow.
        # Therefore we have two val dataloaders, where one is the same as the train dataloader.
        # Thats why we get a warning for validation dataloder that we should switch off shuffling.
        self.named_val_loaders = {"val": val_dataloader, "val_train": train_dataloader}
        self.named_test_loaders = {
            "test_pdbbind2020": test_dataloader_pdbbind2020,
            "test_coach420": test_dataloader_coach420,
            "test_holo4k": test_dataloader_holo4k,
        }

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return list(self.named_val_loaders.values())

    def test_dataloader(self):
        return list(self.named_test_loaders.values())
