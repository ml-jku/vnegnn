# coding=utf-8
# Adapted from Equipocket
import os
from pathlib import Path
from typing import List

import click
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from rdkit import Chem
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, knn
from tqdm import tqdm

"""
This file is used to get the geometric and multi-level structure info
in a protein for our EquiPocket.
Pls intall msms(https://ccsb.scripps.edu/msms/) first.
"""


class MoleculeFeatures(object):
    def __init__(self, file_name):
        self.file_name = file_name
        if file_name[-3:] == "pdb":
            self.molecule = Chem.MolFromPDBFile(file_name, sanitize=False)
        if file_name[-4:] == "mol2":
            self.molecule = Chem.MolFromMol2File(file_name, sanitize=False)
        if file_name[-3:] == "sdf":
            self.molecule = Chem.SDMolSupplier(file_name, sanitize=False)[0]

    # get_bond_length
    def get_bond_length(self, x_0, y_0, z_0, x_1, y_1, z_1):
        bond_length = ((x_0 - x_1) ** 2 + (y_0 - y_1) ** 2 + (z_0 - z_1) ** 2) ** 0.5
        return bond_length

    # get atom features
    def get_atom_features(self, tmp_atom, cal_sasa=False):
        tmp_data = []
        atom_index = tmp_atom.GetIdx()
        tmp_data.append(tmp_atom.GetAtomicNum())
        tmp_data.append(tmp_atom.GetFormalCharge() + 2)
        chiral_tag_list = 0
        chiral_tag = str(tmp_atom.GetChiralTag())
        if chiral_tag == "CHI_UNSPECIFIED":
            pass
        if chiral_tag == "CHI_TETRAHEDRAL_CW":
            chiral_tag_list = 1
        if chiral_tag == "CHI_TETRAHEDRAL_CCW":
            chiral_tag_list = 2
        if chiral_tag == "CHI_OTHER":
            chiral_tag_list = 3
        tmp_data.append(chiral_tag_list)
        tmp_data.append(1 if tmp_atom.GetIsAromatic() else 0)
        # Check if atom is in ring
        tmp_data.append(1 if tmp_atom.IsInRing() else 0)
        tmp_data.append(tmp_atom.GetDegree())
        x, y, z = self.molecule.GetConformer().GetAtomPosition(atom_index)
        pos = (x, y, z)
        return atom_index, tmp_data, pos

    # get edge_feature
    def get_edge_features(self, tmp_bond):
        tmp_result = []
        start_index = tmp_bond.GetBeginAtomIdx()
        end_index = tmp_bond.GetEndAtomIdx()
        # SINGLE, AROMATIC, DOUBLE, Zero
        bond_type = str(tmp_bond.GetBondType())
        bond_type_list = 0
        if bond_type == "SINGLE":
            bond_type_list = 1
        if bond_type == "DOUBLE":
            bond_type_list = 2
        if bond_type == "AROMATIC":
            bond_type_list = 3
        bond_ring = 1 if tmp_bond.IsInRing() else 0
        x_0, y_0, z_0 = self.molecule.GetConformer().GetAtomPosition(start_index)
        x_1, y_1, z_1 = self.molecule.GetConformer().GetAtomPosition(end_index)
        bond_length = self.get_bond_length(x_0, y_0, z_1, x_1, y_1, z_1)
        tmp_result = []
        tmp_result.append(bond_type_list)
        tmp_result += [bond_ring, bond_length]
        return start_index, end_index, tmp_result

    # regard molecule as graph
    def get_graph_features(self, init_index=0):
        self.all_atoms = {}
        atoms = self.molecule.GetAtoms()
        all_atom_index = []
        all_atom_features = []
        all_atom_pos = []
        for tmp_atom in atoms:
            atom_index, atom_feature, pos = self.get_atom_features(tmp_atom)
            all_atom_index.append(init_index + atom_index)
            all_atom_features.append(atom_feature)
            all_atom_pos.append(pos)
        bonds = self.molecule.GetBonds()
        all_edge_index = [[], []]
        all_edge_attr = []
        for tmp_bond in bonds:
            start_index, end_index, edge_feature = self.get_edge_features(tmp_bond)
            all_edge_index[0].append(init_index + start_index)
            all_edge_index[1].append(init_index + end_index)
            all_edge_attr.append(edge_feature)
            all_edge_index[1].append(init_index + start_index)
            all_edge_index[0].append(init_index + end_index)
            all_edge_attr.append(edge_feature)
        return (
            all_atom_index,
            all_atom_features,
            all_atom_pos,
            all_edge_index,
            all_edge_attr,
        )

    # get_surface_feature from msms
    def get_surface(self, msms_path="", output_dir=None):
        vert_surface = []
        pdb_file = self.file_name
        pdb_file = os.path.abspath(pdb_file)

        # Use output_dir if provided, otherwise use dir of pdb file
        if output_dir is None:
            output_dir = os.path.dirname(pdb_file)

        # Create temporary working directory
        work_dir = output_dir
        os.makedirs(work_dir, exist_ok=True)

        # Set output paths in the work directory
        xyzr_path = os.path.join(work_dir, "tmp.xyzr")
        result_prefix = os.path.join(work_dir, "result")

        run_shell = "cd %s ;" % msms_path
        run_shell += "pdb_to_xyzr %s > %s;" % (pdb_file, xyzr_path)
        run_shell += "msms -probe_radius 1.5 -if %s -af %s -of %s" % (
            xyzr_path,
            result_prefix,
            result_prefix,
        )
        run_result = os.system(run_shell)

        # 0: success
        if run_result == 0:
            result_vert_path = result_prefix + ".vert"

            # get vert
            tmp_i = 0
            f = open(result_vert_path)
            for line in f:
                tmp_i += 1
                if tmp_i <= 3:
                    continue
                line = list(map(float, line.strip().split()))
                vert_surface.append(line)
            f.close()

            # Clean up temporary xyzr file
            if os.path.exists(xyzr_path):
                os.remove(xyzr_path)
        else:
            raise RuntimeError(f"MSMS failed for {pdb_file}")

        return vert_surface


def get_surface_feature(vert_surface, protein_pos, mean_protein_pos):
    pos = protein_pos
    vert_pos = vert_surface[:, [0, 1, 2]]
    vert_pos = torch.unique(vert_pos, dim=0)
    vert_pos = vert_pos - mean_protein_pos
    dist_atom_pos_vert_pos = torch.cdist(vert_pos.clone(), pos)
    vert_atom = torch.argmin(dist_atom_pos_vert_pos, dim=1)
    vert_atom = vert_atom.long()
    atom_in_surface = torch.zeros(protein_pos.shape[0])
    atom_in_surface[vert_atom.unique().long()] = 1
    vert_atom_diff = vert_pos - pos[vert_atom]
    vert_num = torch.tensor(vert_atom.shape[0])
    sort_vert_atom, indices = torch.sort(vert_atom)
    vert_atom = sort_vert_atom
    vert_pos = vert_pos[indices]
    vert_atom_diff = vert_atom_diff[indices]
    vert_surface = vert_surface[indices]
    _, vert_batch = torch.unique(vert_atom, return_inverse=True)
    return (vert_pos, vert_atom, vert_num, atom_in_surface, vert_atom_diff, vert_batch)


def get_surface_descriptor(pos, vert_pos, vert_atom, atom_in_surface, vert_batch):
    # KNN for two nearest surface point
    assign_index = knn(vert_pos, vert_pos, 3)
    edge_0 = assign_index[0]
    edge_1 = assign_index[1]
    mask_edge = edge_0 == edge_1
    edge_0 = edge_0[~mask_edge]
    edge_0 = vert_pos[edge_0]
    edge_1 = edge_1[~mask_edge]
    edge_1 = vert_pos[edge_1]
    edge_diff = edge_0 - edge_1
    edge_diff = edge_diff.view(vert_pos.shape[0], 2, 3)
    length_edge_0 = edge_diff[:, 0, :].norm(dim=1).unsqueeze(dim=-1)
    length_edge_1 = edge_diff[:, 1, :].norm(dim=1).unsqueeze(dim=-1)
    angle_knn = (
        torch.mul(F.normalize(edge_diff[:, 0, :]), F.normalize(edge_diff[:, 1, :]))
        .sum(dim=1)
        .unsqueeze(dim=-1)
    )
    angle_knn[torch.isnan(angle_knn)] = 1
    # the former 3 features for local geometric
    knn_geometric_feature = torch.concat(
        [length_edge_0, length_edge_1, angle_knn], dim=1
    )
    # the latter 4 features
    surface_center_pos = global_mean_pool(vert_pos, vert_batch)
    surface_pos_to_center = vert_pos - surface_center_pos[vert_batch]
    surface_pos_to_atom = vert_pos - pos[vert_atom]
    surface_center_to_atom = surface_center_pos - pos[atom_in_surface == 1]
    dist_atom_to_surface_center = (
        surface_center_to_atom.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    )
    dist_surface_point_to_surface_center = (
        surface_pos_to_center.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    )
    dist_surface_point_to_atom = (
        surface_pos_to_atom.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    )
    cos_surface_point_atom = (
        torch.mul(surface_pos_to_center, surface_center_to_atom[vert_batch])
        .sum(dim=1)
        .unsqueeze(dim=-1)
    )
    cos_surface_point_atom = cos_surface_point_atom / (
        dist_atom_to_surface_center[vert_batch]
    )
    cos_surface_point_atom = (
        cos_surface_point_atom / dist_surface_point_to_surface_center
    )
    cos_surface_point_atom[torch.isnan(cos_surface_point_atom)] = 1
    surface_shape_geometric_feature = torch.concat(
        [
            dist_surface_point_to_surface_center,
            dist_surface_point_to_atom,
            dist_atom_to_surface_center[vert_batch],
            cos_surface_point_atom,
        ],
        dim=1,
    )
    surface_descriptor = torch.concat(
        [knn_geometric_feature, surface_shape_geometric_feature], dim=1
    )
    return surface_descriptor, surface_center_pos


def get_protein_feature(protein_file_name, msms_path="", output_dir=None):
    protein = MoleculeFeatures(protein_file_name)
    # get global structure features
    (all_atom_index, all_atom_features, all_atom_pos, all_edge_index, all_edge_attr) = (
        protein.get_graph_features()
    )
    all_atom_pos = torch.tensor(all_atom_pos).float()
    mean_protein_pos = all_atom_pos.mean(dim=0)
    all_atom_pos = all_atom_pos - mean_protein_pos
    # get_surface_features
    vert_surface = protein.get_surface(msms_path=msms_path, output_dir=output_dir)
    vert_surface = tensor(vert_surface).float()
    (vert_pos, vert_atom, vert_num, atom_in_surface, vert_atom_diff, vert_batch) = (
        get_surface_feature(vert_surface, all_atom_pos, mean_protein_pos)
    )
    # get_surface_descriptor
    surface_descriptor, surface_center_pos = get_surface_descriptor(
        all_atom_pos, vert_pos, vert_atom, atom_in_surface, vert_batch
    )
    # trans data -> graph data
    all_atom_features = tensor(all_atom_features).float()
    all_edge_index = tensor(all_edge_index)
    all_edge_attr = tensor(all_edge_attr).float()
    graph_data = Data(
        x=all_atom_features,
        pos=all_atom_pos,
        edge_index=all_edge_index,
        edge_attr=all_edge_attr,
        atom_in_surface=atom_in_surface,
        vert_surface=vert_surface,
        vert_pos=vert_pos,
        vert_atom=vert_atom,
        vert_num=vert_atom,
        vert_atom_diff=vert_atom_diff,
        vert_batch=vert_batch,
        surface_center_pos=surface_center_pos,
        surface_descriptor=surface_descriptor,
        mean_pos=mean_protein_pos,
    )
    return graph_data


def process_single_protein(
    protein_dir: Path, msms_path: str, verbose: bool = False
) -> dict:
    """
    Process a single protein directory.

    Args:
        protein_dir: Path to directory containing protein.pdb
        msms_path: Path to MSMS executable directory
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with status and error information
    """
    result = {"protein_dir": str(protein_dir), "status": "success", "error": None}

    try:
        protein_file = protein_dir / "protein.pdb"

        if not protein_file.exists():
            result["status"] = "skipped"
            result["error"] = f"protein.pdb not found in {protein_dir}"
            return result

        # Process the protein
        graph_data = get_protein_feature(
            str(protein_file), msms_path=msms_path, output_dir=str(protein_dir)
        )

        # Save the graph data
        output_file = protein_dir / "protein_graph.pt"
        torch.save(graph_data, output_file)

        if verbose:
            print(f"✓ Processed {protein_dir.name}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        if verbose:
            print(f"✗ Failed {protein_dir.name}: {e}")

    return result


def find_protein_directories(input_path: Path) -> List[Path]:
    """
    Find all directories containing protein.pdb files.

    Args:
        input_path: Root path to search

    Returns:
        List of directories containing protein.pdb
    """
    protein_dirs = []

    if input_path.is_file():
        # If input is a file listing directories
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    path = Path(line)
                    if path.exists() and (path / "protein.pdb").exists():
                        protein_dirs.append(path)
    else:
        # If input is a directory, search recursively
        for pdb_file in input_path.rglob("protein.pdb"):
            protein_dirs.append(pdb_file.parent)

    return sorted(protein_dirs)


@click.command()
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help=(
        "Input path (directory or file listing directories) "
        "containing protein.pdb files"
    ),
)
@click.option(
    "--msms-path",
    "-m",
    type=str,
    default="/system/apps/userenv/sestak/common/bin/msms",
    help="Path to MSMS executable directory",
)
@click.option(
    "--n-jobs",
    "-j",
    type=int,
    default=-1,
    help="Number of parallel jobs (-1 uses all cores)",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Print detailed progress information"
)
def main(input_path: str, msms_path: str, n_jobs: int, verbose: bool):
    """
    Process protein structures to generate graph features using EquiPocket.

    This script finds all directories containing protein.pdb files and
    processes them to generate graph representations with surface
    features using MSMS.
    The output includes:
    - protein_graph.pt: PyTorch geometric graph data
    - result.vert: MSMS vertex file
    - result.area: MSMS area file
    - result.face: MSMS face file

    Example usage:
        python protein_feature.py -i data/sc-pdb/raw -j 8
    """
    click.echo("EquiPocket Protein Feature Extraction")
    click.echo("=" * 50)

    input_path = Path(input_path)

    # Find all protein directories
    click.echo(f"Searching for protein.pdb files in {input_path}...")
    protein_dirs = find_protein_directories(input_path)

    if not protein_dirs:
        click.echo("No protein.pdb files found!", err=True)
        return

    click.echo(f"Found {len(protein_dirs)} protein directories to process")
    click.echo(f"Using MSMS from: {msms_path}")
    click.echo(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all cores'}")
    click.echo("=" * 50 + "\n")

    # Process proteins in parallel
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(process_single_protein)(protein_dir, msms_path, verbose)
        for protein_dir in tqdm(protein_dirs, desc="Processing proteins")
    )

    # Summary
    click.echo("\n" + "=" * 50)
    click.echo("Processing Summary:")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    click.echo(f"  ✓ Success: {success}")
    click.echo(f"  ✗ Failed: {failed}")
    click.echo(f"  ⊘ Skipped: {skipped}")

    # Print failed cases if any
    if failed > 0:
        click.echo("\nFailed cases:")
        for r in results:
            if r["status"] == "failed":
                click.echo(f"  - {r['protein_dir']}: {r['error']}")

    click.echo("=" * 50)


if __name__ == "__main__":
    main()
