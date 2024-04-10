import argparse

import esm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.models.egnn import EGNNClassifierGlobalNodeHetero
from src.utils.common import get_clusterd_predictions
from src.utils.graph import HeteroGraphBuilder
from src.utils.protein import ProteinInfoInference
from src.utils.visualization import PymolSphere, PymolVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict the binding site of a protein")
    parser.add_argument("-i", "--input", type=str, required=True, help="PDB file to predict")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path")
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Checkpoint of the model"
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the binding site"
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="The device to use")
    return parser.parse_args()


def process_pdb_file(path: str, device="cuda:0") -> ProteinInfoInference:
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    protein_info = ProteinInfoInference(path, name="protein1")
    protein_info.create_esm_features(model, batch_converter, device=device)
    return protein_info


def create_graph(protein_info: ProteinInfoInference) -> HeteroData:
    # Default settings for the best model.
    graph_builder: HeteroGraphBuilder = HeteroGraphBuilder(
        protein_info=protein_info,
        neigh_dist_cutoff=10,
        max_neighbours=10,
        number_of_global_nodes=8,
        esm_features=True,
    )
    return graph_builder.create_graph()


def load_model(checkpoint_path: str, device="cuda:0") -> EGNNClassifierGlobalNodeHetero:
    model: EGNNClassifierGlobalNodeHetero = EGNNClassifierGlobalNodeHetero.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=device, strict=False
    )
    return model


def create_visualization(
    pos_global_node: np.array, ranks_global_node: np.array, protein_path: str, output_path: str
):
    visualizer: PymolVisualizer = PymolVisualizer(
        protein_path=protein_path, protein_color="#2DD4BF"
    )
    for i, (pos, rank) in enumerate(zip(pos_global_node, ranks_global_node)):
        visualizer.add_cgo(
            PymolSphere(position=pos.tolist(), name=f"Score_{rank.item():.3f}", color="#C084FC")
        )

    visualizer.create_visualization(f"{output_path}/visualization.pse")


def predict(model: EGNNClassifierGlobalNodeHetero, graph: HeteroData):
    with torch.no_grad():
        _, pos_global_node, _, ranks_global_node = model(graph)
    pos_global_node = pos_global_node.cpu().numpy()
    ranks_global_node = ranks_global_node.cpu().numpy()

    pos_global_node, ranks_global_node = get_clusterd_predictions(
        pos_sample=pos_global_node, rank_sample=ranks_global_node
    )
    return pos_global_node, ranks_global_node


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output_path
    device = args.device
    checkpoint = args.checkpoint
    visualize = args.visualize

    protein_info: ProteinInfoInference = process_pdb_file(path=input_path, device=device)
    graph: HeteroData = create_graph(protein_info=protein_info).to(device=device)

    model: EGNNClassifierGlobalNodeHetero = load_model(checkpoint_path=checkpoint, device=device)
    model.eval()

    pos_global_node, ranks_global_node = predict(model=model, graph=graph)

    combined_array = np.hstack((pos_global_node, ranks_global_node))
    df = pd.DataFrame(combined_array, columns=["x", "y", "z", "rank"])
    df.to_csv(f"{output_path}/prediction.csv", index=False)

    if visualize:
        create_visualization(
            pos_global_node=pos_global_node,
            ranks_global_node=ranks_global_node,
            protein_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
