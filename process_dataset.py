# preprocess pdb file
import argparse
import itertools
import os

import esm
import lmdb
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


def process_chunk(chunk, data_path, device="cpu"):
    # Initialize model for each process
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()  # Disables dropout

    batch_converter = alphabet.get_batch_converter()

    batch_protein_info = []
    for complex in chunk:
        try:
            complex_path = os.path.join(data_path, complex)
            protein_info = ProteinInfo(complex_path, name=complex)
            protein_info.create_esm_features(model, batch_converter, device)
            batch_protein_info.append(protein_info)
        except ValueError as e:
            print(e)
            continue
        except AttributeError:
            print(f"Error in {complex}")
            continue
        except FileNotFoundError:
            print(f"File not found in: {complex}")
            continue
    return batch_protein_info


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    device = args.device
    n_jobs = args.n_jobs

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
