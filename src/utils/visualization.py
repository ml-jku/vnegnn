from abc import ABC, abstractmethod
from typing import List

from pymol import cmd
from pymol.cgo import COLOR, CONE, CYLINDER, SPHERE


def parse_selection(selection: List):
    parsed_selection = []
    for s in selection:
        chain, atom_name, res_number = s.split(":")
        parsed_selection.append(f"chain {chain} and name {atom_name} and resi {res_number}")

    return " or ".join(parsed_selection)


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.strip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return [r, g, b]


def hex_to_pymol_color(hex_color: str):
    r, g, b = hex_to_rgb(hex_color)
    return [r / 255.0, g / 255.0, b / 255.0]


class PymolCGO(ABC):
    def __init__(self, name: str, color: str):
        self.name = name
        self._color = hex_to_pymol_color(color)

    @property
    def color(self):
        return self._color

    @abstractmethod
    def get_cgo(self):
        pass


class PymolSphere(PymolCGO):
    def __init__(self, position: List, name: str, color: str):
        super().__init__(name, color)
        self.position = position

    def get_cgo(self):
        color = self.color
        return [COLOR, *color, SPHERE, *self.position, 1.0]


class PymolArrows(PymolCGO):
    def __init__(
        self,
        arrows,  #: List[List[List[float, float, float], List[float, float, float]]],
        name: str,
        color: str,
    ):
        super().__init__(name, color)
        self.arrows = arrows  # List of tuples containing start and end positions for each arrow

    def get_cgo(self):
        cgo = []
        color = self.color
        for start, end in self.arrows:
            # Add the cylinder (shaft of the arrow)
            cgo += [CYLINDER, *start, *end, 0.2, *color, *color]
            # Add the cone (arrowhead)
            # Adjust the cone's base and height as needed
            # cone_base = [end[i] + 0.1 * (end[i] - start[i]) for i in range(3)]
            # cgo += [CONE, *end, *cone_base, 0.4, 0.0, *color, *color, 1.0, 0.0]
        return cgo


class PymolSelection:
    def __init__(self, selection: List, name: str, color: str, type: str = "surface"):
        self.selection = selection
        self.name = name
        self._color = hex_to_pymol_color(color)
        self.type = type

    @property
    def color(self):
        return self._color

    def get_selection(self):
        selection_parsed = parse_selection(self.selection)
        return selection_parsed


class PymolVisualizer:
    def __init__(
        self,
        protein_path,
        protein_color="wheat",
        ligand_paths=[],
        ligand_color=None,
        vis_type="surface",
    ):
        self.protein_path: str = protein_path
        self.protein_color: str = protein_color
        self.ligand_paths: List[str] = ligand_paths
        self.ligand_color: str = ligand_color
        self.vis_type: str = vis_type
        self.cgos: List[PymolCGO] = []
        self.selections: List[PymolSelection] = []

    def add_cgo(self, cgo: PymolCGO):
        self.cgos.append(cgo)

    def add_selection(self, selection: PymolSelection):
        self.selections.append(selection)

    def _load_ligands(self):
        # TODO: There is an error it not using # before color, fix it.
        if self.ligand_color is not None:
            if self.ligand_color.startswith("#"):
                cmd.set_color("ligand_color", hex_to_pymol_color(self.ligand_color))
                self.ligand_color: str = "ligand_color"

        for i, ligand_path in enumerate(self.ligand_paths):
            cmd.load(ligand_path, f"ligand_{i}")
            if self.ligand_color is not None:
                cmd.color("ligand_color", f"ligand_{i}")

    def _load_cgos(self):
        for pymol_cgo in self.cgos:
            cmd.load_cgo(pymol_cgo.get_cgo(), pymol_cgo.name)

    def _load_selections(self):
        for pymol_selection in self.selections:
            selection = pymol_selection.get_selection()
            if len(selection) == 0:
                continue
            cmd.select("selection", selection)
            cmd.show_as(pymol_selection.type, "selection")
            cmd.create(pymol_selection.name, "selection")
            cmd.set_color(f"{pymol_selection.name}_color", pymol_selection.color)
            cmd.color(f"{pymol_selection.name}_color", pymol_selection.name)
            cmd.delete("selection")

    def create_visualization(self, export_path: str = "./export.pse"):
        cmd.reinitialize()
        if self.protein_color.startswith("#"):
            cmd.set_color("protein_color", hex_to_pymol_color(self.protein_color))
            self.protein_color: str = "protein_color"

        cmd.load(self.protein_path, "protein")
        cmd.show_as(self.vis_type, "protein")
        cmd.color("protein_color", "protein")

        self._load_selections()
        self._load_ligands()
        self._load_cgos()

        cmd.deselect()
        cmd.save(export_path)
