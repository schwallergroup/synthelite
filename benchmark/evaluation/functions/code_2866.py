from typing import Tuple, Dict, List
import copy
from rdkit.Chem import AllChem, rdFMCS
from collections import deque
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
import rdkit.Chem.rdFMCS
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem import rdmolops
import re
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, Descriptors
import traceback
import rdkit
from collections import Counter


from rdkit import Chem

# Refactoring for Enumeration: Isolate the list of motifs
MOTIFS_TO_PRESERVE = {
    "chlorobenzene": Chem.MolFromSmarts("[#6]1:[#6]:[#6](-[#17]):[#6]:[#6]:[#6]:1"),
    "isoxazole": Chem.MolFromSmarts("[#6]1:[#7]:[#8]:[#6]:[#6]:1"),
    "morpholine": Chem.MolFromSmarts("[#7]1-[#6]-[#6]-[#8]-[#6]-[#6]-1"),
}

def main(route) -> Tuple[bool, Dict]:
    """
    Detects if at least one specified structural motif is preserved in every product
    throughout the entire synthesis. The motifs are defined in the MOTIFS_TO_PRESERVE
    list and include chlorobenzene, isoxazole, and morpholine.
    """
    findings_template = {
        "atomic_checks": {
            "named_reactions": [],
            "ring_systems": [],
            "functional_groups": []
        },
        "structural_constraints": []
    }
    findings_json = copy.deepcopy(findings_template)

    # Use the predefined motifs
    motifs = MOTIFS_TO_PRESERVE

    # Track which motifs are present at each step
    motifs_present = {name: [] for name in motifs}
    all_products = []

    def dfs_traverse(node, depth=0):
        nonlocal findings_json, motifs_present, all_products
        if node["type"] == "reaction":
            # Extract product
            rsmi = node["metadata"]["mapped_reaction_smiles"]
            product_smiles = rsmi.split(">")[-1]
            product = Chem.MolFromSmiles(product_smiles) if product_smiles else None

            if product:
                all_products.append(product)
                # Check each motif
                for name, pattern in motifs.items():
                    if product.HasSubstructMatch(pattern):
                        motifs_present[name].append(True)
                        # Record atomic check for ring system
                        if name not in findings_json["atomic_checks"]["ring_systems"]:
                            findings_json["atomic_checks"]["ring_systems"].append(name)
                    else:
                        motifs_present[name].append(False)

        # Determine the next depth based on the current node's type
        next_depth = depth
        if node["type"] != "reaction":  # If current node is 'chemical'
            next_depth = depth + 1

        # Continue traversing
        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    # Start traversal
    dfs_traverse(route)

    # Check if we have products and if motifs are preserved
    if not all_products:
        return False, findings_json

    preserved_motifs = []
    for name, presence_list in motifs_present.items():
        if presence_list and all(presence_list):
            preserved_motifs.append(name)

    # Strategy is present if at least one motif is preserved
    strategy_present = len(preserved_motifs) >= 1

    if strategy_present:
        # Add the structural constraint if the strategy is present
        findings_json["structural_constraints"].append({
            "type": "count",
            "details": {
                "target": "preserved_motif",
                "operator": ">=",
                "value": 1
            }
        })

    return strategy_present, findings_json
