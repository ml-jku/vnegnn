#!/usr/bin/env python3
"""
Command-line tool for extracting atom-level binding info from protein ligand complexes.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import rootutils
from Bio.PDB import PDBParser
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


def extract_binding_atoms(
    protein_info: ProteinInfo,
    ligand_infos: List[LigandInfo],
    threshold: float = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract binding atoms information from protein and ligand data.

    Returns:
        binding_atoms: Boolean array indicating which atoms are binding atoms
        binding_site_centers: Center coordinates of each binding site
    """
    binding_atoms_ligands = []
    binding_site_centers = []

    for ligand_info in ligand_infos:
        # Calculate distances between all protein atoms and ligand atoms
        dists = np.linalg.norm(
            protein_info.coords[:, None] - ligand_info.coords[None], axis=-1
        )
        # Mark atoms that are within threshold distance to any ligand atom
        below_threshold = (dists <= threshold).any(axis=1)

        binding_atoms_coords = protein_info.coords[below_threshold]
        binding_site_center = binding_atoms_coords.mean(axis=0)

        binding_atoms_ligands.append(below_threshold)
        binding_site_centers.append(binding_site_center)

    # Combine binding atoms from all ligands (OR operation)
    return (
        np.array(binding_atoms_ligands).any(axis=0),
        np.array(binding_site_centers),
    )


def process_single_complex(complex_path: Path, threshold: float = 4) -> bool:
    """Process a single protein-ligand complex and extract atom-level binding information."""
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
        binding_atoms, binding_site_centers = extract_binding_atoms(
            protein_info, ligand_infos, threshold
        )

        # Create one-hot vector for binding atoms (1 = binding, 0 = non-binding)
        binding_atoms_one_hot = binding_atoms.astype(np.int32)

        # Collect ligand information
        lig_ids, lig_coords = [], []
        for i, ligand_info in enumerate(ligand_infos):
            lig_ids.extend(len(ligand_info.coords) * [i])
            lig_coords.append(ligand_info.coords)
        lig_ids = np.array(lig_ids)
        lig_coords = np.concatenate(lig_coords, axis=0)

        # Save atom-level binding information
        output_path = complex_path / "binding_atoms.npz"
        np.savez(
            str(output_path),
            atom_coords=protein_info.coords,
            atom_names=protein_info.atom_names,
            atom_types=protein_info.atom_types,
            res_names=protein_info.res_names,
            res_ids=protein_info.res_ids,
            chains=protein_info.chains,
            binding_atoms=binding_atoms_one_hot,
            binding_site_centers=binding_site_centers,
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
def extract_binding_atoms_cli(
    path: Path, n_jobs: int, threshold: float, verbose: bool, backend: str
):
    """
    Extract atom-level binding information from protein-ligand complexes.

    This script processes directories containing protein-ligand complexes,
    extracts atom-level binding site information, and saves the results as
    binding_atoms.npz files.

    Each complex directory should contain:
    - protein.pdb: Protein structure file
    - ligand_*.sdf/mol2/pdb: Ligand structure files

    The output binding_atoms.npz contains:
    - atom_coords: Coordinates of all protein atoms
    - atom_names: Names of all atoms (e.g., CA, N, C, O)
    - atom_types: Element types of all atoms
    - res_names: Residue names for each atom
    - res_ids: Residue IDs for each atom
    - chains: Chain IDs for each atom
    - binding_atoms: One-hot vector (1=binding, 0=non-binding)
    - binding_site_centers: Centers of binding sites
    - ligand_coords: Coordinates of ligand atoms
    - ligand_ids: Ligand IDs for each ligand atom
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
    extract_binding_atoms_cli()
