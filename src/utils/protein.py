import logging
import os
import shutil
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)


def clean_pdb_file(path):
    """
    Cleans the PDB file in-place, keeping only residue information and removing
    hydrogen atoms, water molecules, and heteroatoms.
    The cleaned content is written back to the same input file, overwriting it.

    Parameters:
    path (str): Path to the PDB file to be cleaned.
    """
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp()
    with os.fdopen(temp_fd, "w") as temp_file:
        with open(path, "r") as input_file:
            for line in input_file:
                # Keep only "ATOM" records and exclude hydrogen atoms
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    # Skip hydrogen atoms, which start with 'H' or 'D' (deuterium)
                    if atom_name.startswith("H") or atom_name.startswith("D"):
                        continue
                    # Write the non-hydrogen "ATOM" records to the temporary file
                    temp_file.write(line)

    # Replace the original file with the cleaned temporary file
    shutil.move(temp_path, path)


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Load a molecule from a file of format
    ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.
    https://github.com/HannesStark/EquiBind

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format
        ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ""
        for line in pdbqt_data:
            pdb_block += "{}\n".format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError(
            "Expect the format of the molecule_file to be "
            "one of .mol2, .sdf, .pdbqt and .pdb, got {}".format(molecule_file)
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except Exception:
                log.info("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception:
        return None

    return mol
