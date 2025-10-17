#!/usr/bin/env python3
"""
Command-line tool for extracting binding info from protein ligand complexes.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import click
import numpy as np
import rootutils
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from rdkit.Chem import Mol
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.protein import read_molecule  # noqa: E402


@dataclass
class ProteinInfo:
    coords: np.ndarray
    res_names: np.ndarray[str]
    atom_names: np.ndarray[str]
    atom_types: np.ndarray[str]
    res_ids: np.ndarray[int]
    chains: np.ndarray[str]


@dataclass
class LigandInfo:
    coords: np.ndarray


def extract_ligand_info(ligand: Mol) -> LigandInfo:
    """Extract ligand information from RDKit molecule."""
    conf = ligand.GetConformer()
    positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()]
    coords = np.array(
        [[ligand_point.x, ligand_point.y, ligand_point.z] for ligand_point in positions]
    )
    return LigandInfo(coords=coords)


def extract_protein_info(protein: Structure) -> ProteinInfo:
    """Extract protein information from BioPython structure."""
    coords = []
    res_names = []
    atom_names = []
    atom_types = []
    res_ids = []
    chains = []

    for model in protein:
        for chain in model:
            for residue in chain:
                # Only process standard residues (not HETATM)
                if residue.id[0] != " ":
                    continue
                # Skip non-standard amino acids that seq1() can't convert
                # This ensures consistency with ESM embedding generation
                try:
                    seq1(residue.resname)
                except KeyError:
                    continue
                for atom in residue:
                    coords.append(atom.get_coord())
                    res_names.append(residue.resname)
                    atom_names.append(atom.name)
                    atom_types.append(atom.element)
                    res_ids.append(residue.id[1])
                    chains.append(chain.id)

    return ProteinInfo(
        coords=np.array(coords),
        res_names=np.array(res_names),
        atom_names=np.array(atom_names),
        atom_types=np.array(atom_types),
        res_ids=np.array(res_ids),
        chains=np.array(chains),
    )


def calculate_residue_depths(protein: Structure) -> Optional[dict]:
    """
    Calculate residue depth for all residues in the protein structure.

    Returns a dictionary mapping (chain_id, residue_id) to residue depth,
    or None if MSMS is not available or calculation fails.
    """
    try:
        model = protein[0]
        rd = ResidueDepth(model)

        # Create a mapping of (chain_id, res_id) -> depth
        depth_map = {}
        for chain in model:
            for residue in chain:
                # residue.id is a tuple: (hetero_flag, res_id, insertion_code)
                res_id = residue.id[1]
                chain_id = chain.id
                key = (chain_id, res_id)

                try:
                    # ResidueDepth returns (residue_depth, ca_depth)
                    depth_tuple = rd[chain_id, residue.id]
                    depth_map[key] = depth_tuple[0]  # Use residue depth
                except KeyError:
                    # Some residues might not have depth calculated
                    depth_map[key] = np.nan

        return depth_map
    except Exception as e:
        # MSMS might not be installed or other errors
        click.echo(f"Warning: Could not calculate residue depths: {str(e)}")
        click.echo("Make sure MSMS is installed and in your PATH")
        return None


def extract_binding_site(
    protein_info: ProteinInfo,
    ligand_infos: List[LigandInfo],
    threshold: float = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract binding site information from protein and ligand data."""
    binding_atoms_ligands = []
    binding_site_centers = []

    for ligand_info in ligand_infos:
        dists = np.linalg.norm(
            protein_info.coords[:, None] - ligand_info.coords[None], axis=-1
        )
        below_threshold = (dists <= threshold).any(axis=1)

        binding_atoms_coords = protein_info.coords[below_threshold]
        binding_site_center = binding_atoms_coords.mean(axis=0)

        binding_atoms_ligands.append(below_threshold)
        binding_site_centers.append(binding_site_center)

    return (
        np.array(binding_atoms_ligands).any(axis=0),
        np.array(binding_site_centers),
    )


def process_single_complex(complex_path: Path, threshold: float = 4) -> bool:
    """Process a single protein-ligand complex and extract binding information."""
    try:
        protein_path = complex_path / "protein.pdb"
        if not protein_path.exists():
            click.echo(f"Warning: {protein_path} not found, skipping {complex_path}")
            return False
        protein = PDBParser().get_structure("protein", str(protein_path))
        protein_info = extract_protein_info(protein)

        ligand_paths = [
            complex_path / f for f in os.listdir(complex_path) if f.startswith("ligand")
        ]
        if not ligand_paths:
            click.echo(f"Warning: No ligand files found in {complex_path}")
            return False

        ligand_mols = []
        for ligand_path in ligand_paths:
            mol = read_molecule(str(ligand_path))
            if mol is not None:
                ligand_mols.append(mol)
            else:
                click.echo(f"Warning: Could not load ligand {ligand_path}")

        if not ligand_mols:
            click.echo(f"Warning: No valid ligands found in {complex_path}")
            return False

        ligand_infos = [extract_ligand_info(ligand) for ligand in ligand_mols]

        # Extract binding site information based on atoms being within a threshold
        # distance of a ligand atom.
        binding_atoms, binding_site_centers = extract_binding_site(
            protein_info, ligand_infos, threshold
        )

        # Calculate residue depths
        depth_map = calculate_residue_depths(protein)

        # Extract only CA atoms as used in VN-EGNN as input.
        ca_atoms = protein_info.atom_names == "CA"
        binding_residues = (binding_atoms & ca_atoms)[ca_atoms]

        res_coords = protein_info.coords[ca_atoms]
        res_names = protein_info.res_names[ca_atoms]
        res_ids = protein_info.res_ids[ca_atoms]
        chains = protein_info.chains[ca_atoms]

        # Get residue depths for CA atoms
        if depth_map is not None:
            res_depths = np.array(
                [
                    depth_map.get((chain_id, res_id), np.nan)
                    for chain_id, res_id in zip(chains, res_ids)
                ]
            )
        else:
            # If depth calculation failed, set all to NaN
            res_depths = np.full(len(res_ids), np.nan)

        lig_ids, lig_coords = [], []
        for i, ligand_info in enumerate(ligand_infos):
            lig_ids.extend(len(ligand_info.coords) * [i])
            lig_coords.append(ligand_info.coords)
        lig_ids = np.array(lig_ids)
        lig_coords = np.concatenate(lig_coords, axis=0)

        output_path = complex_path / "binding.npz"
        np.savez(
            str(output_path),
            binding_residues=binding_residues,
            binding_site_centers=binding_site_centers,
            res_coords=res_coords,
            res_names=res_names,
            res_ids=res_ids,
            chains=chains,
            res_depths=res_depths,
            ligand_coords=lig_coords,
            ligand_ids=lig_ids,
        )

        click.echo(f"Successfully processed {complex_path}")
        return True

    except Exception as e:
        click.echo(f"Error processing {complex_path}: {str(e)}")
        return False


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to directory containing protein-ligand complex folders",
)
@click.option(
    "--n-jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs to run",
)
@click.option(
    "--threshold",
    "-t",
    default=4.0,
    type=float,
    help="Distance threshold for binding site detection (Angstroms)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--backend",
    "-b",
    default="processes",
    type=click.Choice(["threads", "processes"]),
    help="Backend to use for parallel processing",
)
def extract_binding_info(
    path: Path, n_jobs: int, threshold: float, verbose: bool, backend: str
):
    """
    Extract binding information from protein-ligand complexes.

    This script processes directories containing protein-ligand complexes,
    extracts binding site information, and saves the results as binding.npz files.

    Each complex directory should contain:
    - protein.pdb: Protein structure file
    - ligand_*.sdf/mol2/pdb: Ligand structure files
    """
    if verbose:
        click.echo(f"Processing complexes in: {path}")
        click.echo(f"Number of parallel jobs: {n_jobs}")
        click.echo(f"Binding site threshold: {threshold} Å")

    complex_dirs = [d for d in path.iterdir() if d.is_dir()]
    if not complex_dirs:
        click.echo(f"No directories found in {path}")
        return

    if verbose:
        click.echo(f"Found {len(complex_dirs)} complex directories")

    if n_jobs == 1:
        results = []
        for complex_dir in tqdm(complex_dirs, desc="Processing complexes"):
            result = process_single_complex(complex_dir, threshold)
            results.append(result)
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, prefer=backend)(
            delayed(process_single_complex)(complex_dir, threshold)
            for complex_dir in tqdm(complex_dirs, desc="Processing complexes")
        )

    successful = sum(results)
    total = len(results)
    click.echo(
        f"\nProcessing complete: {successful}/{total} complexes processed successfully"
    )


if __name__ == "__main__":
    extract_binding_info()
