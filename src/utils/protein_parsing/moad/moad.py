import os
import shutil

import requests

from src.utils.protein_parsing.moad.lfetch import extractLigand


def parse_ligand(lig):
    filename = lig.replace(" ", "_")
    names = lig.split(":")[0].split()
    chain = lig.split(":")[1]
    lig_dict = {
        "filename": filename,
        "name(s)": names,
        "chain": chain,
        "resid(s)": [str(i) for i in range(int(lig.split(":")[2]), int(lig.split(":")[2]) + len(names))],
    }
    return lig_dict


def download_pdb(pdb_code, filename, biological_assembly=False):
    if not filename.endswith(".pdb"):
        filename = filename + ".pdb"
    if biological_assembly:
        url = "https://files.rcsb.org/download/" + pdb_code + ".pdb1"
    else:
        url = "https://files.rcsb.org/download/" + pdb_code + ".pdb"
    r = requests.get(url, allow_redirects=True)

    with open(filename, "wb") as f:
        f.write(r.content)


def extract_pdb_ligand_from_p2rank_datasets(pdb_code, src_path, dst_path, df_moad):
    try:
        # This is needed because of the format of the coach420 dataset, where the pdb code is 5
        # characters long, because it contains only a certain chain
        full_pdb_code = pdb_code[:4] if len(pdb_code) == 5 else pdb_code
        entry = df_moad.loc[lambda x: (x["pdb_code"] == full_pdb_code) & (x["validity"] == "valid")]
        ligand_list = []
        for lig in entry["ligand"].tolist():
            ligand_list.append(parse_ligand(lig))

        for lig in ligand_list:
            dirname = f"{os.path.join(dst_path, pdb_code)}_{lig['filename']}"
            src_pdb_filename = os.path.join(src_path, f"{pdb_code}.pdb")
            dst_pdb_filename = os.path.join(dirname, "protein.pdb")
            src_lig_file_name = os.path.join(dirname, f"{lig['filename']}.pdb")
            dst_lig_file_name = os.path.join(dirname, "ligand.pdb")
            pdb_filename_clean = os.path.join(src_path, f"{pdb_code}_clean.pdb")

            os.mkdir(dirname)
            extractLigand(src_pdb_filename, [lig], dirname, False, False, True)

            shutil.move(pdb_filename_clean, dst_pdb_filename)
            # Check if ligand file is exists, because MOAD database is based on all chains
            # and the ligand might not be present in the chain
            if not os.path.exists(src_lig_file_name) or os.stat(src_lig_file_name).st_size == 0:
                shutil.rmtree(dirname)
                continue

            os.rename(src_lig_file_name, dst_lig_file_name)

    except Exception as e:
        print(f"Error in {pdb_code}: {e}")
        return pdb_code


def extract_pdb_ligand(pdb_code, save_path, df_moad, biological_assembly=False):
    try:
        entry = df_moad.loc[lambda x: (x["pdb_code"] == pdb_code) & (x["validity"] == "valid")]
        ligand_list = []
        for lig in entry["ligand"].tolist():
            ligand_list.append(parse_ligand(lig))

        for lig in ligand_list:
            dirname = f"{os.path.join(save_path, pdb_code)}_{lig['filename']}"
            os.mkdir(dirname)
            pdb_filename = os.path.join(dirname, "protein.pdb")
            download_pdb(pdb_code, pdb_filename, biological_assembly)
            extractLigand(pdb_filename, [lig], dirname, False, False, True)
            os.remove(pdb_filename)
            pdb_filename_clean = os.path.join(dirname, "protein_clean.pdb")
            os.rename(pdb_filename_clean, pdb_filename)
            ligand_file_name = os.path.join(dirname, f"{lig['filename']}.pdb")
            os.rename(ligand_file_name, os.path.join(dirname, "ligand.pdb"))

            # Check if ligand file is empty, this can happend because of
            # biological assemblies and ligand present in asymetric unit
            if os.stat(os.path.join(dirname, "ligand.pdb")).st_size == 0:
                shutil.rmtree(dirname)

    except Exception as e:
        print(e)
        print(pdb_code)
        return pdb_code
