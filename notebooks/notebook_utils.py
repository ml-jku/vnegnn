from pathlib import Path

import nglview as nv
import pandas as pd
from nglview.shape import Shape


def render_protein(protein_name: str, df: pd.DataFrame, dataset_path: Path):
    complex_path = Path(f"{dataset_path}/{protein_name}")
    pdb_file = complex_path / "protein.pdb"
    ligands = [lig for lig in complex_path.glob("ligand_*.pdb")]
    if len(ligands) == 0:
        ligands = [lig for lig in complex_path.glob("ligand_*.mol2")]

    view = nv.show_file(pdb_file.as_posix())
    view.layout.width = "800px"
    view.layout.height = "600px"

    view.clear_representations()
    view.add_representation(
        "surface", selection="protein", opacity=0.3, color="lightblue"
    )  # noqa: E501
    view.add_representation("cartoon", selection="protein", color="secondary structure")

    for ligand in ligands:
        view.add_component(ligand.as_posix())
        view.add_representation("ball+stick", component=view.n_components - 1)

    shape = Shape(view)

    p = df.loc[lambda x: x["protein_name"] == protein_name].reset_index(drop=True)
    for i, row in p.iterrows():
        confidence = (
            round(row["confidence_0"], 3) if hasattr(row, "confidence_0") else "n"
        )
        shape.add_sphere(
            [row["x"], row["y"], row["z"]], [1, 0, 0], 1.5, f"{str(confidence)}"
        )  # red

    if hasattr(row, "vn_initial_pos_0") and row["vn_initial_pos_0"] is not None:
        for i, row in p.iterrows():
            shape.add_sphere(
                [
                    row["vn_initial_pos_0"],
                    row["vn_initial_pos_1"],
                    row["vn_initial_pos_2"],
                ],
                [0, 1, 0],
                1,
                "vn_initial_pos",
            )

    return view
