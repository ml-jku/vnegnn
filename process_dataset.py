# preprocess pdb file
import argparse
import itertools
import os

import lmdb
from openbabel import openbabel as ob
from tqdm.notebook import tqdm

from src.utils.common import pmap_multi
from src.utils.protein import ProteinInfo, serialize


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PDB files")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input PDB file")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to the output PDB file"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for ESM")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run")
    return parser.parse_args()


def convert_mol2_to_pdb(input_file, output_file):
    try:
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("mol2", "pdb")
        mol = ob.OBMol()
        obConversion.ReadFile(mol, input_file)
        obConversion.WriteFile(mol, output_file)
        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")


def process_chunk(chunk, data_path, device="cpu"):
    # Initialize model for each process

    batch_protein_info = []
    for complex in chunk:
        try:
            complex_path = os.path.join(data_path, complex)
            protein_pdb_path = os.path.join(complex_path, "protein.pdb")
            if not os.path.exists(protein_pdb_path):
                protein_mol2_path = os.path.join(complex_path, "protein.mol2")
                convert_mol2_to_pdb(protein_mol2_path, protein_pdb_path)
            protein_info = ProteinInfo(complex_path, name=complex)
            batch_protein_info.append(protein_info)
        except ValueError as e:
            print(e)
            continue
        except AttributeError:
            print(f"Error in {complex}")
            continue
        except FileNotFoundError:
            print(f"File not found in: {complex_path}")
            continue
        except Exception as e:
            print(f"Error in {complex}: {e}")
            continue
    return batch_protein_info


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    device = args.device
    n_jobs = args.n_jobs

    os.makedirs(output_path, exist_ok=True)

    complexes = os.listdir(input_path)
    # Split complexes into chunks
    n = 100
    chunks = [(complexes[i : i + n],) for i in range(0, len(complexes), n)]
    result = pmap_multi(process_chunk, chunks, n_jobs=n_jobs, data_path=input_path, device=device)
    result = list(itertools.chain(*result))

    save_path = os.path.join(output_path, "protein_info.lmdb")
    map_size = 171798691840
    env = lmdb.open(save_path, create=True, map_size=map_size)
    with env.begin(write=True) as txn:
        for proteinInfo in tqdm(result):
            serialized_obj = serialize(proteinInfo)
            txn.put(proteinInfo.name.encode(), serialized_obj)

    env.close()


if __name__ == "__main__":
    main()
