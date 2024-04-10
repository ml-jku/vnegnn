import logging
import os
import os.path
import pickle
import shutil
import tempfile
from typing import List, Optional

import esm
import numpy as np
import pandas as pd
import torch
from Bio import PDB
from Bio.PDB import PDBParser
from openbabel import openbabel
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)


def clean_pdb_file(path):
    """
    Cleans the PDB file in-place, keeping only residue information and removing hydrogen atoms, water molecules, and heteroatoms.
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
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.
    https://github.com/HannesStark/EquiBind

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
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
            except:
                log.info("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol


def load_ligand(lig_path):
    lig = read_molecule(f"{lig_path}.sdf", sanitize=True, remove_hs=False)
    if lig is None:
        lig = read_molecule(f"{lig_path}.mol2", sanitize=True, remove_hs=False)
    if lig is None:
        lig = read_molecule(f"{lig_path}.pdb", sanitize=True, remove_hs=False)
    if lig is None:
        log.warning(f"Ligand: [{lig_path}] can't be parsed by rdkit.")
        return None

    return lig


def convert_format(input_file, input_format, output_file, output_format):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(input_format, output_format)
    mol = openbabel.OBMol()
    success = obConversion.ReadFile(mol, input_file)
    if success:
        obConversion.WriteFile(mol, output_file)
        return True
    return False


class PDBProcessor:
    """This class is used to extract ligands from a pdb file.
    Used to process the https://github.com/rdk/p2rank-datasets (testsets: coach420, holo4k).
    These datasets have the ligand inside the pdb file, and a list of relevant ligands is available in
    the repository. This class extracts the ligands from the pdb file and saves them in the desired format.
    It also saves the protein structure without the ligands and water atoms.

    Example:

    processor = PDBProcessor("path_to_pdb.pdb")

    ligand_files = processor.extract_ligands(["HDA", "IMP", "GPX"], desired_formats=["sdf", "mol2"])

    protein_file = processor.save_protein_structure()

        Args:
            pdb_file_path (str): The path to the pdb file, to extract the ligands from.
    """

    def __init__(self, pdb_file_path: str):
        self.pdb_file_path = pdb_file_path
        parser = PDB.PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", pdb_file_path)

    def extract_ligands(self, ligand_names, desired_formats=["pdb"], output_dir=""):
        ligands = []
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    # Check that it is a HETATM entry and the name matches the desired ligand names
                    if residue.id[0] != " " and resname in ligand_names:
                        ligands.append(residue.copy())  # Add a copy of the residue

        ligand_files = {}
        io = PDB.PDBIO()
        for i, residue in enumerate(ligands):
            s = PDB.Structure.Structure(residue.get_resname())
            m = PDB.Model.Model(0)
            s.add(m)
            c = PDB.Chain.Chain("A")
            m.add(c)
            c.add(residue)

            ligand_file_name = f"ligand{i+1}_{residue.get_resname()}"
            pdb_output_path = os.path.join(output_dir, f"{ligand_file_name}.pdb")
            io.set_structure(s)
            io.save(pdb_output_path)

            # Convert the pdb file to each desired format and store the paths
            format_files = {}
            for fmt in desired_formats:
                if fmt != "pdb":
                    output_path = os.path.join(output_dir, f"{ligand_file_name}.{fmt}")
                    conversion_success = convert_format(pdb_output_path, "pdb", output_path, fmt)
                    if conversion_success:
                        format_files[fmt] = output_path
                    else:
                        print(f"Warning: Failed to convert {pdb_output_path} to {fmt}")

            # If "pdb" is not in desired formats, remove the pdb file
            if "pdb" not in desired_formats:
                os.remove(pdb_output_path)
            else:
                format_files["pdb"] = pdb_output_path

            ligand_files[ligand_file_name] = format_files

        return ligand_files

    def save_protein_structure(self, output_file_name="protein.pdb"):
        io = PDB.PDBIO()

        # Copy the original structure to avoid modifying it
        protein_structure = self.structure.copy()

        # Remove ligands and water atoms
        for model in protein_structure:
            for chain in model:
                residues_to_remove = [
                    residue
                    for residue in chain
                    if residue.id[0] != " " or residue.get_resname() == "HOH"
                ]
                for residue in residues_to_remove:
                    chain.detach_child(residue.id)

        io.set_structure(protein_structure)
        io.save(output_file_name)

        return output_file_name


class BindingSiteExtractor:
    def __init__(self, protein, ligand):
        """"""
        self.protein = protein
        self.ligand = ligand

    def get_ligand_coords(self) -> np.ndarray:
        """Get the coordinates of the ligand.

        Returns:
            np.ndarray: The coordinates of the ligand.
        """
        conf = self.ligand.GetConformer()
        ligand_points = [conf.GetAtomPosition(atom.GetIdx()) for atom in self.ligand.GetAtoms()]
        ligand_coords = np.array(
            [[ligand_point.x, ligand_point.y, ligand_point.z] for ligand_point in ligand_points]
        )
        return ligand_coords

    def calculate_nearest_atom_distance(self) -> pd.DataFrame:
        """Calculate the distance from an atom on the protein to the nearest atom on the ligand.

        Returns:
            pd.DataFrame: The dataframe containing the distance from an atom on the protein to the
            nearest atom on the ligand.
        """
        ligand_coords = self.get_ligand_coords()
        data = []

        for chain in self.protein:
            for residue in chain:
                for atom in residue:
                    atom_coords = atom.get_coord()

                    distances = [
                        np.linalg.norm(atom_coords - ligand_coord)
                        for ligand_coord in ligand_coords
                    ]
                    min_distance = min(distances)
                    element = atom.element
                    chain = atom.full_id[2]
                    atom_name = atom.full_id[4][0]

                    data.append(
                        [
                            element,
                            atom_coords[0],
                            atom_coords[1],
                            atom_coords[2],
                            min_distance,
                            residue.resname,
                            residue.id[1],
                            chain,
                            atom_name,
                        ]
                    )

        df = pd.DataFrame(
            data,
            columns=[
                "element",
                "x",
                "y",
                "z",
                "min_distance_to_ligand",
                "res",
                "res_number",
                "chain",
                "atom_name",
            ],
        )
        return df


class Complex:
    def __init__(
        self,
        complex_path: str,
        bindingsite_cutoff: float = 4,
        surface_cutoff: Optional[float] = None,
        name: str = None,
    ):
        """Complex class for loading a protein complex. Contains the protein, ligands
        and the surface. The surface is calculated using BioPython's ResidueDepth module.

            Args:
                complex_path (str): _description_
                bindingsite_cutoff (float, optional): _description_. Defaults to 4.
                surface_cutoff (Optional[float], optional): _description_. Defaults to None.
                name (str, optional): _description_. Defaults to None.
        """
        self.complex_path = complex_path
        self.bindingsite_cutoff = bindingsite_cutoff
        self.surface_cutoff = surface_cutoff
        self.name = name
        self._surface = None
        self._df_atoms = None
        self._protein = None

        self.import_protein()
        self.import_ligands()

    def import_protein(self):
        filename = f"{self.complex_path}/protein.pdb"
        parser = PDBParser()
        self._structure = parser.get_structure("protein", filename)
        self._protein = self._structure[0]
        self._pocket_center = None

    def import_ligands(self):
        files_in_path = os.listdir(self.complex_path)
        ligand_files = [f for f in files_in_path if f.startswith("ligand")]
        distinct_ligands = set([f.split(".")[0] for f in ligand_files])
        self._ligands = {l: self.load_ligand(l) for l in distinct_ligands}

    def load_ligand(self, ligand_name: str):
        lig_path = f"{self.complex_path}/{ligand_name}"

        lig = read_molecule(f"{lig_path}.sdf", sanitize=True, remove_hs=False)
        if lig is None:
            lig = read_molecule(f"{lig_path}.mol2", sanitize=True, remove_hs=False)
        if lig is None:
            lig = read_molecule(f"{lig_path}.pdb", sanitize=True, remove_hs=False)
        if lig is None:
            log.warning(f"Ligand: [{lig_path}] can't be parsed by rdkit.")
            return None

        return lig

    def get_binding_site_atoms_for_ligand(self, ligand_name):
        if ligand_name not in self._ligands:
            raise ValueError(f"Ligand {ligand_name} is not part of this complex.")

        ligand = self._ligands[ligand_name]
        binding_site_extractor = BindingSiteExtractor(self._protein, ligand)
        nearest_atom_distances = binding_site_extractor.calculate_nearest_atom_distance()
        return nearest_atom_distances

    def calculate_atoms(self):
        all_bindingsite_atoms = []

        for ligand_name in self._ligands:
            ligand = self._ligands[ligand_name]
            binding_site_extractor = BindingSiteExtractor(self._protein, ligand)
            df_atoms = binding_site_extractor.calculate_nearest_atom_distance()
            bindingsite_atoms = df_atoms["min_distance_to_ligand"] <= self.bindingsite_cutoff
            all_bindingsite_atoms.append(bindingsite_atoms)

        all_bindingsite_atoms = np.array(all_bindingsite_atoms).T.sum(axis=1)
        df_atoms["bindingsite"] = all_bindingsite_atoms > 0
        df_atoms = df_atoms.drop(columns=["min_distance_to_ligand"])

        self._pocket_center = df_atoms.loc[
            lambda x: x["bindingsite"] == True, ["x", "y", "z"]
        ].mean()

        return df_atoms.reset_index(drop=True)

    @property
    def pocket_center(self):
        if self._pocket_center is None:
            self.calculate_atoms()
        return self._pocket_center

    @property
    def protein_atoms(self):
        if self._df_atoms is None:
            self._df_atoms = self.calculate_atoms()
        return self._df_atoms

    @property
    def ligand_coords(self):
        assert len(self._ligands) == 1, "Only one ligand supported currently."
        ligand = self._ligands[list(self._ligands.keys())[0]]
        conf = ligand.GetConformer()
        ligand_points = [conf.GetAtomPosition(atom.GetIdx()) for atom in ligand.GetAtoms()]
        ligand_coords = np.array(
            [[ligand_point.x, ligand_point.y, ligand_point.z] for ligand_point in ligand_points]
        )
        return ligand_coords


three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def get_protein_structure(res_list):
    "https://github.com/QizhiPei/FABind"
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [
        res
        for res in res_list
        if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))
    ]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure["name"] = "placeholder"
    structure["seq"] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res["N"], res["CA"], res["C"], res["O"]]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure["coords"] = coords
    return structure


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    "https://github.com/QizhiPei/FABind"
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == " ":
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ("CA" in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res["CA"].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def extract_protein_structure(path):
    "https://github.com/QizhiPei/FABind"
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", path)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    sturcture = get_protein_structure(res_list)
    return sturcture


def extract_esm_feature(protein, device="cuda:3"):
    "https://github.com/QizhiPei/FABind"
    device = device if torch.cuda.is_available() else "cpu"

    letter_to_num = {
        "C": 4,
        "D": 3,
        "S": 15,
        "Q": 5,
        "K": 11,
        "I": 9,
        "P": 14,
        "T": 16,
        "F": 13,
        "A": 0,
        "G": 7,
        "H": 8,
        "E": 6,
        "L": 10,
        "R": 1,
        "W": 17,
        "V": 19,
        "N": 2,
        "Y": 18,
        "M": 12,
    }

    # Load ESM-2 model with different sizes
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", protein["seq"]),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33][0][1:-1]
    assert token_representations.shape[0] == len(protein["seq"])
    return token_representations


class ProteinInfo:
    def __init__(self, path: str, name: str):
        self.path: str = path
        self.name: str = name

        self.chains: List[str] = None
        self.coordiantes: List[np.array] = None
        self.res_names: List[str] = None
        self.seq: str = None
        self.atom_names: List[str] = None
        self.res_numbers: List[int] = None
        self.res_features: List[np.array] = None
        self.res_binding: List[bool] = None
        self.pocket_center: np.array = None
        self.ligand_coords: np.array = None

        self.create_info()

    def create_info(self):
        complex = Complex(self.path, name=self.name)

        # Mark a CA as part of BS if any of its atoms is in BS
        df_atoms = complex.calculate_atoms().copy()
        df_binding_residues = (
            df_atoms.groupby(["res_number", "chain"])["bindingsite"].any().reset_index()
        )
        df_atoms = df_atoms.loc[lambda x: x["atom_name"] == "CA", :].reset_index(drop=True)
        df_atoms = df_atoms.merge(df_binding_residues, on=["res_number", "chain"])
        df_atoms = df_atoms.drop("bindingsite_x", axis=1).rename(
            columns={"bindingsite_y": "bindingsite"}
        )

        self.name = complex.name
        self.chains = df_atoms["chain"].tolist()
        self.coordiantes = df_atoms[["x", "y", "z"]].values
        self.res_names = df_atoms["res"].values.tolist()
        self.seq = df_atoms["res"].map(three_to_one).str.cat()
        self.atom_names = df_atoms["atom_name"].values.tolist()
        self.res_numbers = df_atoms["res_number"].values.tolist()
        # self.res_features=esm_features.cpu().numpy().tolist(),
        self.res_binding = df_atoms["bindingsite"].values.tolist()
        self.pocket_center = complex.pocket_center
        self.ligand_coords = complex.ligand_coords

        if not (
            len(self.res_names)
            == len(self.coordiantes)
            == len(self.atom_names)
            == len(self.res_numbers)
            == len(self.res_binding)
            == len(self.chains)
            == len(self.seq)
        ):
            raise ValueError(f"Error in {self.name}, length of all lists should be equal")

    def create_esm_features(self, model, batch_converter, device):
        data = [
            ("protein1", self.seq),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33][0][1:-1]
        assert token_representations.shape[0] == len(self.seq)
        self.res_features = token_representations.cpu().numpy()

        if not len(self.res_features) == len(self.res_binding):
            raise ValueError(f"Error in {self.name}, length of all lists should be equal")


class ProteinInfoInference:
    def __init__(self, path: str, name: str):
        self.path: str = path
        self.name: str = name

        self.chains: List[str] = None
        self.coordiantes: List[np.array] = None
        self.res_names: List[str] = None
        self.seq: str = None
        self.atom_names: List[str] = None
        self.res_numbers: List[int] = None
        self.res_features: List[np.array] = None
        self.res_binding: List[bool] = None
        self.pocket_center: np.array = None
        self.ligand_coords: np.array = None

        self.create_info()

    def create_info(self):
        parser = PDBParser()
        structure = parser.get_structure(self.name, self.path)
        protein = structure[0]

        data = []
        for chain in protein:
            for residue in chain:
                for atom in residue:
                    atom_coords = atom.get_coord()
                    element = atom.element
                    chain = atom.full_id[2]
                    atom_name = atom.full_id[4][0]

                    data.append(
                        [
                            element,
                            atom_coords[0],
                            atom_coords[1],
                            atom_coords[2],
                            residue.resname,
                            residue.id[1],
                            chain,
                            atom_name,
                        ]
                    )

        df = pd.DataFrame(
            data,
            columns=[
                "element",
                "x",
                "y",
                "z",
                "res",
                "res_number",
                "chain",
                "atom_name",
            ],
        )
        df = df.loc[lambda x: x["atom_name"] == "CA", :].reset_index(drop=True)

        self.name = self.name
        self.chains = df["chain"].tolist()
        self.coordiantes = df[["x", "y", "z"]].values
        self.res_names = df["res"].values.tolist()
        self.seq = df["res"].map(three_to_one).str.cat()
        self.atom_names = df["atom_name"].values.tolist()
        self.res_numbers = df["res_number"].values.tolist()

        if not (
            len(self.res_names)
            == len(self.coordiantes)
            == len(self.atom_names)
            == len(self.res_numbers)
            == len(self.chains)
            == len(self.seq)
        ):
            raise ValueError(f"Error in {self.name}, length of all lists should be equal")

    def create_esm_features(self, model, batch_converter, device):
        data = [
            ("protein1", self.seq),
        ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33][0][1:-1]
        assert token_representations.shape[0] == len(self.seq)
        self.res_features = token_representations.cpu().numpy()


def serialize(obj):
    return pickle.dumps(obj)


def deserialize(serialized_obj):
    return pickle.loads(serialized_obj)
