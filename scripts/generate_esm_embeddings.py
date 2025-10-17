#!/usr/bin/env python3
"""
Command-line tool for generating ESM embeddings from protein PDB files.
"""

import os
from pathlib import Path

import click
import esm
import h5py
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def extract_sequence_from_pdb(pdb_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """
    Extract amino acid sequence from PDB file.

    Returns:
        tuple: (concatenated_sequence, list of (chain_id, chain_sequence) tuples)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    chain_sequences = []
    full_sequence = ""
    skipped_residues = []

    for model in structure:
        for chain in model:
            chain_seq = ""
            for residue in chain:
                if residue.id[0] == " ":  # Only standard residues
                    try:
                        aa_3letter = residue.resname
                        aa_1letter = seq1(aa_3letter)

                        # Skip residues without CA atoms
                        # (must match binding info extraction)
                        if "CA" not in residue:
                            skipped_residues.append(
                                f"{chain.id}:{residue.resname}{residue.id[1]}"
                                f" (no CA)"
                            )
                            continue

                        chain_seq += aa_1letter
                    except KeyError:
                        # Non-standard amino acid that seq1() doesn't recognize
                        skipped_residues.append(
                            f"{chain.id}:{residue.resname}{residue.id[1]}"
                        )
                        continue

            if chain_seq:  # Only add non-empty chains
                chain_id = chain.id if chain.id.strip() else "A"
                chain_sequences.append((chain_id, chain_seq))
                full_sequence += chain_seq

    if skipped_residues:
        import warnings

        residue_list = ", ".join(skipped_residues[:5])
        extra_msg = (
            f" and {len(skipped_residues) - 5} more"
            if len(skipped_residues) > 5
            else ""
        )
        warnings.warn(
            f"Skipped {len(skipped_residues)} non-standard residues in "
            f"{pdb_path.name}: {residue_list}{extra_msg}"
        )

    return full_sequence, chain_sequences


def extract_sequences_batch(protein_names: list, base_path: Path) -> dict:
    """Extract sequences from multiple PDB files."""
    sequences = {}
    errors = {}

    for protein_name in protein_names:
        pdb_path = base_path / protein_name / "protein.pdb"
        if not pdb_path.exists():
            errors[protein_name] = "PDB file not found"
            continue

        try:
            full_sequence, chain_sequences = extract_sequence_from_pdb(pdb_path)
            if not full_sequence:
                errors[protein_name] = "No valid sequence extracted"
            else:
                sequences[protein_name] = {
                    "full_sequence": full_sequence,
                    "chain_sequences": chain_sequences,
                }
        except Exception as e:
            errors[protein_name] = f"Error extracting sequence: {str(e)}"

    return sequences, errors


def generate_esm_embeddings_batch(
    protein_names: list,
    base_path: Path,
    model_name: str = "esm2_t33_650M_UR50D",
    output_format: str = "hdf5",
    device: str = "auto",
) -> dict:
    """
    Generate ESM embeddings for a batch of proteins.

    Each chain in multi-chain proteins is processed separately through ESM,
    then the embeddings are concatenated in order to preserve chain boundaries.
    """
    results = {}
    model = None
    alphabet = None
    batch_converter = None

    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        elif device == "cpu":
            model = model.cpu()

        sequences, errors = extract_sequences_batch(protein_names, base_path)

        for protein_name, error in errors.items():
            results[protein_name] = f"Error: {error}"

        if not sequences:
            return results

        # Process each protein separately to handle multi-chain properly
        for protein_name, seq_data in sequences.items():
            try:
                full_sequence = seq_data["full_sequence"]
                chain_sequences = seq_data["chain_sequences"]

                # Process each chain separately
                chain_embeddings = []
                chain_ids = []
                chain_lengths = []

                for chain_id, chain_seq in chain_sequences:
                    # Process single chain
                    data = [(chain_id, chain_seq)]
                    _, _, chain_tokens = batch_converter(data)
                    chain_lens = (chain_tokens != alphabet.padding_idx).sum(1)

                    device_to_use = next(model.parameters()).device
                    chain_tokens = chain_tokens.to(device_to_use)

                    with torch.no_grad():
                        chain_results = model(
                            chain_tokens, repr_layers=[33], return_contacts=False
                        )

                    chain_repr = chain_results["representations"][33]

                    # Extract residue embeddings (skip BOS and EOS tokens)
                    start_idx = 1
                    end_idx = chain_lens[0] - 1

                    chain_emb = chain_repr[0, start_idx:end_idx].cpu().numpy()
                    chain_embeddings.append(chain_emb)
                    chain_ids.append(chain_id)
                    chain_lengths.append(len(chain_seq))

                    # Clean up immediately
                    del chain_results, chain_repr, chain_tokens

                # Concatenate all chain embeddings in order
                residue_embeddings = np.concatenate(chain_embeddings, axis=0)

                # Generate per-sequence representation via averaging
                sequence_embedding = residue_embeddings.mean(axis=0)

                # Save embeddings immediately
                output_path = base_path / protein_name / f"embeddings.{output_format}"

                if output_format == "hdf5":
                    with h5py.File(output_path, "w") as f:
                        f.create_dataset("residue_embeddings", data=residue_embeddings)
                        f.create_dataset("sequence_embedding", data=sequence_embedding)
                        f.create_dataset("sequence", data=full_sequence.encode("utf-8"))
                        f.attrs["model_name"] = model_name
                        f.attrs["sequence_length"] = len(full_sequence)
                        f.attrs["embedding_dim"] = residue_embeddings.shape[1]
                        f.attrs["num_chains"] = len(chain_sequences)

                        # Store chain information
                        chain_ids_encoded = [cid.encode("utf-8") for cid in chain_ids]
                        f.create_dataset("chain_ids", data=chain_ids_encoded)
                        f.create_dataset("chain_lengths", data=chain_lengths)

                elif output_format == "npz":
                    np.savez_compressed(
                        output_path,
                        residue_embeddings=residue_embeddings,
                        sequence_embedding=sequence_embedding,
                        sequence=full_sequence,
                        chain_ids=chain_ids,
                        chain_lengths=chain_lengths,
                        num_chains=len(chain_sequences),
                    )
                else:
                    results[protein_name] = (
                        f"Error: Unsupported output format {output_format}"
                    )
                    continue

                results[protein_name] = (
                    f"Successfully generated embeddings for {protein_name} "
                    f"(seq_len: {len(full_sequence)}, chains: {len(chain_sequences)})"
                )

                # Explicitly delete variables to free memory
                del residue_embeddings
                del sequence_embedding
                del chain_embeddings

            except Exception as e:
                results[protein_name] = f"Error processing {protein_name}: {str(e)}"

        # Clear all large variables
        del sequences

        # Clear GPU memory after processing batch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        # If model loading or batch processing fails, mark all as failed
        for protein_name in protein_names:
            results[protein_name] = f"Error: {str(e)}"

    finally:
        # Ensure all variables are deleted
        if model is not None:
            del model
        if alphabet is not None:
            del alphabet
        if batch_converter is not None:
            del batch_converter

        import gc

        gc.collect()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to the directory containing protein subdirectories",
)
@click.option(
    "--model",
    "-m",
    default="esm2_t33_650M_UR50D",
    help="ESM model to use (default: esm2_t33_650M_UR50D)",
)
@click.option(
    "--output-format",
    "-o",
    default="npz",
    type=click.Choice(["hdf5", "npz"]),
    help="Output format for embeddings (default: hdf5)",
)
@click.option(
    "--batch-size",
    "-b",
    default=1,
    type=int,
    help="Number of proteins to process in each batch (default: 8)",
)
@click.option(
    "--n-jobs",
    "-j",
    default=1,
    type=int,
    help="Number of parallel jobs (default: 1, use 1 for GPU memory efficiency)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--monitor-memory",
    "-m",
    is_flag=True,
    help="Monitor GPU memory usage",
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    type=click.Choice(["auto", "cpu", "cuda"]),
    help="Device to use for computation (default: auto)",
)
def generate_embeddings(
    path: Path,
    model: str,
    output_format: str,
    batch_size: int,
    n_jobs: int,
    verbose: bool,
    monitor_memory: bool,
    device: str,
) -> None:
    """
    Generate ESM embeddings for proteins in PDB format using batched processing.

    This tool processes protein PDB files and generates ESM embeddings for each
    protein. Each subdirectory should contain a file named 'protein.pdb'.

    Multi-chain proteins are handled correctly by processing each chain separately
    through ESM, then concatenating the embeddings in order. This preserves chain
    boundaries and ensures proper context for each chain.

    The tool uses batched processing for efficiency - multiple proteins are
    processed together in each batch, but embeddings are saved individually.

    The output includes:
    - Per-residue embeddings (token-level representations)
    - Per-sequence embeddings (averaged representations)
    - Original amino acid sequence
    - Chain IDs and lengths (for multi-chain proteins)

    Example usage:
        python generate_esm_embeddings.py --path data/proteins
        python generate_esm_embeddings.py -p data/proteins -m esm2_t12_35M_UR50D \\
            -o npz -b 16 -j 1 -v --monitor-memory
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"Generating ESM embeddings using model: {model}")
    click.echo(f"Input path: {path}")
    click.echo(f"Output format: {output_format}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Device: {device}")
    click.echo(f"Using {n_jobs} parallel jobs")

    # Show initial memory usage if monitoring
    if monitor_memory and device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        # memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        click.echo(
            f"Initial GPU Memory - Allocated: {memory_allocated:.2f}GB, "
            "Reserved: {memory_reserved:.2f}GB"
        )

    click.echo("-" * 50)

    try:
        protein_names = [d for d in os.listdir(path) if (path / d).is_dir()]
        if not protein_names:
            click.echo("No subdirectories found in the specified path!")
            return
    except Exception as e:
        click.echo(f"Error reading directory: {e}")
        return

    click.echo(f"Found {len(protein_names)} protein directories")

    # Check for missing PDB files
    missing_files = []
    for protein_name in protein_names:
        pdb_file = path / protein_name / "protein.pdb"
        if not pdb_file.exists():
            missing_files.append(protein_name)

    if missing_files:
        click.echo(f"Warning: {len(missing_files)} directories missing PDB files:")
        for name in missing_files[:5]:  # Show first 5
            click.echo(f"  - {name}")
        if len(missing_files) > 5:
            click.echo(f"  ... and {len(missing_files) - 5} more")
        click.echo()

    # Filter out proteins without PDB files
    valid_proteins = [name for name in protein_names if name not in missing_files]

    if not valid_proteins:
        click.echo("No valid proteins found!")
        return

    click.echo(f"Processing {len(valid_proteins)} proteins with PDB files")

    # Create batches of proteins
    batches = [
        valid_proteins[i : i + batch_size]
        for i in range(0, len(valid_proteins), batch_size)
    ]
    click.echo(f"Created {len(batches)} batches of up to {batch_size} proteins each")

    try:
        all_results = {}

        if n_jobs == 1:
            # Sequential batch processing
            for batch in tqdm(batches, desc="Processing batches"):
                batch_results = generate_esm_embeddings_batch(
                    batch, path, model, output_format, device
                )
                all_results.update(batch_results)
                if verbose:
                    for protein_name, result in batch_results.items():
                        click.echo(f"  {result}")

                # Force garbage collection after each batch
                import gc

                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Monitor memory usage if requested
                if monitor_memory and device == "cuda" and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    # memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    click.echo(
                        f"  GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                        "Reserved: {memory_reserved:.2f}GB"
                        "{memory_reserved:.2f}GB"
                    )

                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
        else:
            batch_results_list = Parallel(n_jobs=n_jobs)(
                delayed(generate_esm_embeddings_batch)(
                    batch, path, model, output_format, device
                )
                for batch in tqdm(batches, desc="Processing batches")
            )

            for batch_results in batch_results_list:
                all_results.update(batch_results)

    except KeyboardInterrupt:
        click.echo("\nEmbedding generation interrupted by user!")
        return
    except Exception as e:
        click.echo(f"Error during processing: {e}")
        return

    results = [all_results[name] for name in valid_proteins]
    successful = [r for r in results if "Successfully" in r]
    failed = [r for r in results if "Error" in r]

    click.echo("\n" + "=" * 50)
    click.echo("EMBEDDING GENERATION COMPLETE!")
    click.echo(f"Successful: {len(successful)}")
    click.echo(f"Failed: {len(failed)}")

    if verbose and successful:
        click.echo("\nSuccessful generations:")
        for success in successful:
            click.echo(f"  ✓ {success}")

    if failed:
        click.echo("\nFailed generations:")
        for failure in failed:
            click.echo(f"  ✗ {failure}")

    if failed:
        click.echo(f"\nWarning: {len(failed)} embedding generations failed!")
        exit(1)
    else:
        click.echo("\nAll embedding generations completed successfully!")
        exit(0)


if __name__ == "__main__":
    generate_embeddings()
