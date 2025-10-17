from typing import Dict

import numpy as np
import pandas as pd
from einops import rearrange
from sklearn.cluster import MeanShift
from torch import Tensor, nn
from tqdm import tqdm

from src.modules.cluster import cluster
from src.utils.graph import sample_fibonacci_grid

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def calc_group_var(pos: Tensor, index: Tensor) -> Tensor:
    out = pos.new_zeros(int(index.max().item()) + 1, pos.shape[-1])
    means = torch.index_reduce(out, 0, index, pos, "mean", include_self=False)
    means_x2 = torch.index_reduce(out, 0, index, pos**2, "mean", include_self=False)
    var = means_x2 - means**2
    return var


def multi_predictions(model: nn.Module, batch: Dict[str, Tensor], num_cycles: int):
    pred_pos_cycles = []
    pred_conf_cycles = []
    batch_global_nodes_cycles = []
    virtual_nodes_initial_pos = []

    for i in range(num_cycles):
        _, pred_pos, _, pred_conf = model(batch)
        batch_global_nodes = batch["global_node"].batch

        pred_pos_cycles.append(pred_pos.cpu())
        pred_conf_cycles.append(pred_conf.cpu())
        batch_global_nodes_cycles.append(batch_global_nodes.cpu())

        centroids = rearrange(batch.centroid, "(s d) -> s d", d=3)
        radii = rearrange(batch.radius, "(s d) -> s d", d=1)

        batch_initial_pos = []
        for s in batch["global_node"].batch.unique():
            centroid = centroids[s]
            radius = radii[s]
            sel_mask = s == batch_global_nodes
            num_points = batch["global_node"].pos[sel_mask].shape[0]
            batch_initial_pos.append(batch["global_node"].pos[sel_mask].cpu())
            batch["global_node"].pos[sel_mask] = sample_fibonacci_grid(
                centroid,
                radius,
                num_points,
                random_rotations=True,
            ).to(batch["global_node"].pos.device)
        virtual_nodes_initial_pos.append(torch.cat(batch_initial_pos))

    pred_pos = torch.cat(pred_pos_cycles).to(batch["global_node"].pos.device)
    pred_conf = torch.cat(pred_conf_cycles).to(batch["global_node"].pos.device)
    batch_global_nodes = torch.cat(batch_global_nodes_cycles).to(
        batch["global_node"].pos.device
    )
    virtual_nodes_initial_pos = torch.cat(virtual_nodes_initial_pos).to(
        batch["global_node"].pos.device
    )
    return pred_pos, pred_conf, batch_global_nodes, virtual_nodes_initial_pos


def _to_numpy(x):
    """Best-effort convert tensors to numpy and leave others as-is."""
    if _HAS_TORCH and hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return x


def _is_scalar_like(x):
    # Treat plain numbers/strings/booleans as scalar-like
    return isinstance(x, (int, float, bool, str, np.number, np.bool_))


def _length_of(item):
    """Return length if item is per-row sequence/array, else None."""
    if _is_scalar_like(item) or item is None:
        return None
    # numpy arrays
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            return None
        return item.shape[0]
    # sequences (list/tuple)
    if isinstance(item, (list, tuple)):
        return len(item)
    # pandas Series
    if isinstance(item, pd.Series):
        return len(item)
    return None


def predictions_to_df(predictions):
    """
    Convert nested list of prediction dicts (e.g., Lightning .predict output)
    into a flat pandas DataFrame without hard-coding field names.

    - One row per item.
    - Per-prediction scalars are broadcast to all rows for that prediction.
    - 1D per-item arrays -> one column.
    - 2D per-item arrays -> split into multiple columns: <key>_0, <key>_1, ...
      Special case: if key in {'coords','coord','xyz'} and dim==3, use x,y,z.
    """
    rows = []

    for batch in predictions:
        for pred in batch:
            # Normalize all values first
            norm = {k: _to_numpy(v) for k, v in pred.items()}

            # Determine row count N from any sequence/array value
            candidate_lengths = [_length_of(v) for v in norm.values()]
            candidate_lengths = [L for L in candidate_lengths if L is not None]
            if not candidate_lengths:
                # No per-item fields; treat as a single-row prediction
                N = 1
            else:
                # Validate that all per-item fields agree on N
                N = candidate_lengths[0]
                for L in candidate_lengths[1:]:
                    if L != N:
                        raise ValueError(
                            f"Inconsistent per-item lengths "
                            f"in prediction: {candidate_lengths}"
                        )

            # Prepare per-item values (broadcast scalars)
            # Build column fragments for each key
            per_item_columns = {}  # key -> list/np array of length N (1D)
            multi_col_blocks = {}  # key -> 2D array (N, D)

            for key, val in norm.items():
                L = _length_of(val)

                if L is None:
                    # scalar-like or None -> broadcast
                    if _is_scalar_like(val) or val is None:
                        per_item_columns[key] = [val] * N
                    else:
                        # Unknown type: store repr and broadcast
                        per_item_columns[key] = [repr(val)] * N
                    continue

                # numpy arrays
                if isinstance(val, np.ndarray):
                    if val.ndim == 1:
                        # shape: (N,)
                        if val.shape[0] != N:
                            raise ValueError(
                                f"Field '{key}' length {val.shape[0]} != {N}"
                            )
                        per_item_columns[key] = val
                    elif val.ndim == 2:
                        # shape: (N, D)
                        if val.shape[0] != N:
                            raise ValueError(
                                f"Field '{key}' length {val.shape[0]} != {N}"
                            )
                        multi_col_blocks[key] = val
                    else:
                        # Higher dims -> flatten last dims
                        flat = val.reshape(val.shape[0], -1)
                        if flat.shape[0] != N:
                            raise ValueError(
                                f"Field '{key}' length {flat.shape[0]} != {N}"
                            )
                        multi_col_blocks[key] = flat
                    continue

                # lists/tuples
                if isinstance(val, (list, tuple)):
                    # List of scalars -> 1D
                    if len(val) == 0:
                        per_item_columns[key] = [None] * N
                    else:
                        first = val[0]
                        if _is_scalar_like(first) or first is None:
                            if len(val) != N:
                                raise ValueError(
                                    f"Field '{key}' length {len(val)} != {N}"
                                )
                            per_item_columns[key] = list(val)
                        else:
                            # List of vectors -> try to convert to 2D array
                            arr = np.array(val)
                            if arr.ndim == 1:
                                # ragged; fallback to string repr
                                per_item_columns[key] = [repr(v) for v in val]
                            else:
                                if arr.shape[0] != N:
                                    raise ValueError(
                                        f"Field '{key}' length {arr.shape[0]} != {N}"
                                    )
                                multi_col_blocks[key] = arr
                    continue

                # pandas Series
                if isinstance(val, pd.Series):
                    if len(val) != N:
                        raise ValueError(f"Field '{key}' length {len(val)} != {N}")
                    per_item_columns[key] = val.values
                    continue

                # Fallback: repr per item
                per_item_columns[key] = [repr(v) for v in val]

            # Build rows
            # Start with 1D columns
            base = {
                k: (list(v) if isinstance(v, np.ndarray) else v)
                for k, v in per_item_columns.items()
            }

            # Expand multi-column blocks
            for key, block in multi_col_blocks.items():
                block = np.asarray(block)
                d = block.shape[1] if block.ndim >= 2 else 1

                # Friendly naming for 3D coordinates
                if key.lower() in {"coords", "coord", "xyz"} and d == 3:
                    base["x"] = block[:, 0].astype(float).tolist()
                    base["y"] = block[:, 1].astype(float).tolist()
                    base["z"] = block[:, 2].astype(float).tolist()
                else:
                    for j in range(d):
                        base[f"{key}_{j}"] = block[:, j].tolist()

            # Now create N row dicts
            for i in range(N):
                row = {
                    col: (vals[i] if isinstance(vals, list) else vals[i])
                    for col, vals in base.items()
                }
                # Cast floatable numeric arrays to float where sensible
                for col, val in list(row.items()):
                    if isinstance(val, (np.floating, np.integer, np.bool_)):
                        row[col] = val.item()
                rows.append(row)

    return pd.DataFrame(rows)


def evaluate_protein_predictions(
    protein_name,
    df,
    protein_path,
    num_global_nodes=8,
    threshold=4.0,
    cluster_preds=True,
    cluster_algorithm=MeanShift(),
):
    """
    Evaluate predictions for a single protein.

    Args:
        protein_name: Name of the protein to evaluate
        df: DataFrame with predictions containing columns: protein_name, x, y, z,
            confidence_0, etc.
        protein_path: Path to directory containing protein binding data
        num_global_nodes: Number of top-ranked predictions to evaluate (default: 8)
        threshold: Distance threshold in Angstroms for successful
            prediction (default: 4.0)
        cluster_preds: Whether to cluster predictions before evaluation (default: True)
        cluster_algorithm: Clustering algorithm instance (e.g., MeanShift()),
            required if cluster_preds=True

    Returns:
        dict: Dictionary containing protein_name, num_ligs, and rank metrics
            (n_rank_dca_*, n_rank_dcc_*)
    """
    binding = np.load(protein_path / f"{protein_name}/binding.npz")
    bindingsite_centers = binding["binding_site_centers"]

    df_selected_protein = df.loc[lambda x: x["protein_name"] == protein_name]
    pred_coords = df_selected_protein[["x", "y", "z"]].values
    confs = df_selected_protein["confidence_0"].values

    lig_ids = np.unique(binding["ligand_ids"])
    num_ligs = len(lig_ids)

    n_rank = {f"n_rank_dca_{i}": 0 for i in range(num_global_nodes)}
    if cluster_preds:
        if cluster_algorithm is None:
            raise ValueError(
                "cluster_algorithm must be provided when cluster_preds=True"
            )
        pred_coords, confs = cluster(pred_coords, confs, algorithm=cluster_algorithm)

    confs_rank = np.argsort(confs)[::-1]

    for lig_id in lig_ids:
        lig_coords = binding["ligand_coords"][binding["ligand_ids"] == lig_id]
        for i, rank in enumerate(range(num_global_nodes)):
            top_i = confs_rank[: i + num_ligs]
            top_i_coords = pred_coords[top_i]

            dist = np.linalg.norm(lig_coords[:, None] - top_i_coords, axis=-1)
            below_dist = (dist <= threshold).any()
            n_rank[f"n_rank_dca_{i}"] += 1 if below_dist else 0

    n_rank_dcc = {f"n_rank_dcc_{i}": 0 for i in range(num_global_nodes)}
    for center in bindingsite_centers:
        for i, rank in enumerate(range(num_global_nodes)):
            top_i = confs_rank[: i + num_ligs]
            top_i_coords = pred_coords[top_i]

            dist = np.linalg.norm(center - top_i_coords, axis=-1)
            below_dist = (dist <= threshold).any()
            n_rank_dcc[f"n_rank_dcc_{i}"] += 1 if below_dist else 0

    return {"protein_name": protein_name, "num_ligs": num_ligs, **n_rank, **n_rank_dcc}


def evaluate_all_proteins(
    df,
    protein_path,
    num_global_nodes=8,
    threshold=4.0,
    cluster_preds=True,
    cluster_algorithm=MeanShift(),
    show_progress=True,
):
    """
    Evaluate predictions for all proteins in the dataframe.

    Args:
        df: DataFrame with predictions containing columns: protein_name, x, y, z,
            confidence_0, etc.
        protein_path: Path to directory containing protein binding data
        num_global_nodes: Number of top-ranked predictions to evaluate (default: 8)
        threshold: Distance threshold in Angstroms for successful prediction
            (default: 4.0)
        cluster_preds: Whether to cluster predictions before evaluation (default: True)
        cluster_algorithm: Clustering algorithm instance (e.g., MeanShift()),
            required if cluster_preds=True
        show_progress: Whether to show progress bar (default: True)

    Returns:
        list: List of dictionaries containing evaluation results for each protein
    """
    protein_names = df["protein_name"].unique()
    iterator = tqdm(protein_names) if show_progress else protein_names

    res = []
    for protein_name in iterator:
        result = evaluate_protein_predictions(
            protein_name=protein_name,
            df=df,
            protein_path=protein_path,
            num_global_nodes=num_global_nodes,
            threshold=threshold,
            cluster_preds=cluster_preds,
            cluster_algorithm=cluster_algorithm,
        )
        res.append(result)

    return res


def compute_metric_ratios(df, metric_name, denominator_col="num_ligs", decimals=2):
    metric_cols = [c for c in df.columns if metric_name in c and "ratio" not in c]
    for col in metric_cols:
        ratio_col_name = f"{metric_name}_ratio_{col}"
        df[ratio_col_name] = round(df[col] / df[denominator_col], decimals)
    ratio_cols = [c for c in df.columns if f"{metric_name}_ratio_" in c]
    df_filtered = df[["protein_name", denominator_col] + ratio_cols]
    return df, df_filtered, ratio_cols
