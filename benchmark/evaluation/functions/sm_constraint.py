from functions.utils import flip_reactions, get_route_len
from rdkit import Chem

def get_leaf_molecules(route: dict) -> list:
    """
    Get all the leaf molecules in the route.
    """
    leaf_molecules = []

    def dfs_traverse(node: dict, depth: int):
        if node["type"] == "mol" and not node.get("children", []):
            leaf_molecules.append(node["smiles"])
        elif node["type"] == "mol":
            for child in node["children"]:
                dfs_traverse(child, depth + 1)
        elif node["type"] == "reaction":
            for child in node["children"]:
                dfs_traverse(child, depth + 1)

    dfs_traverse(route, 0)
    return leaf_molecules

def get_all_molecules(route: dict) -> list:
    """
    Get all the leaf molecules in the route.
    """
    res = []

    def dfs_traverse(node: dict, depth: int):
        if node["type"] == "mol":
            res.append(node["smiles"])
            if node.get("children", []):
                for child in node["children"]:
                    dfs_traverse(child, depth + 1)
        elif node["type"] == "reaction":
            for child in node["children"]:
                dfs_traverse(child, depth + 1)

    dfs_traverse(route, 0)
    return res

def check_starting_material(route: dict, sm: str) -> bool:
    """
    Check if the route has a starting material with the given SMILES.
    """
    sm_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(sm))
    return any(Chem.MolToSmiles(Chem.MolFromSmiles(leaf)) == sm_canonical for leaf in get_all_molecules(route))