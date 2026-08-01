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
from functions.sr_utils.check import Check
from functions.sr_utils import fuzzy_dict, check

import os as _os; root_data = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")

fg_args = {
    "file_path": f"{root_data}/patterns/functional_groups.json",
    "value_field": "pattern",
    "key_field": "name",
}
reaction_class_args = {
    "file_path": f"{root_data}/patterns/smirks.json",
    "value_field": "smirks",
    "key_field": "name",
}
ring_smiles_args = {
    "file_path": f"{root_data}/patterns/chemical_rings_smiles.json",
    "value_field": "smiles",
    "key_field": "name",
}
functional_groups = fuzzy_dict.FuzzyDict.from_json(**fg_args)
reaction_classes = fuzzy_dict.FuzzyDict.from_json(**reaction_class_args)
ring_smiles = fuzzy_dict.FuzzyDict.from_json(**ring_smiles_args)

checker = check.Check(
    fg_dict=functional_groups, reaction_dict=reaction_classes, ring_dict=ring_smiles
)

def main(route) -> Tuple[bool, Dict]:
    """
    Detects if a methylsulfonyl group is preserved throughout the synthesis.

    Returns True if the methylsulfonyl group is present in the target molecule
    and is preserved in at least one intermediate (non-starting material).
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

    # Track if methylsulfonyl is in the target molecule
    target_has_methylsulfonyl = False
    # Track if methylsulfonyl is in at least one intermediate
    intermediate_has_methylsulfonyl = False
    # Track the number of intermediates checked
    intermediates_checked = 0

    def dfs_traverse(node, depth=0):
        nonlocal target_has_methylsulfonyl, intermediate_has_methylsulfonyl, intermediates_checked, findings_json

        if node.get("type") == "mol" and "smiles" in node:
            # Skip in_stock materials (starting materials)
            if node.get("in_stock", False):
                return

            mol = Chem.MolFromSmiles(node["smiles"])
            if mol:
                # Check specifically for methylsulfonyl attached to carbon (methyl-SO2-C)
                methylsulfonyl_pattern = Chem.MolFromSmarts("[CH3][S](=[O])(=[O])[#6]")
                has_methylsulfonyl = mol.HasSubstructMatch(methylsulfonyl_pattern)

                if has_methylsulfonyl:
                    findings_json["atomic_checks"]["functional_groups"].append("methylsulfonyl")
                    # If this is the target molecule (depth 0)
                    if depth == 0:
                        target_has_methylsulfonyl = True
                    else:
                        # This is an intermediate
                        intermediate_has_methylsulfonyl = True
                        intermediates_checked += 1

        # Determine the new depth for the recursive call
        new_depth = depth
        if node.get("type") != "reaction": # If current node is chemical, depth increases
            new_depth = depth + 1

        # Continue traversing
        for child in node.get("children", []):
            dfs_traverse(child, new_depth)

    # Start traversal
    dfs_traverse(route)

    # Determine the final result
    result = (
        target_has_methylsulfonyl and intermediate_has_methylsulfonyl and intermediates_checked >= 1
    )

    # Add structural constraints to findings_json if conditions are met
    if target_has_methylsulfonyl:
        findings_json["structural_constraints"].append({
            "type": "positional",
            "details": {
                "target": "methylsulfonyl",
                "position": "last_stage"
            }
        })
    if intermediate_has_methylsulfonyl and intermediates_checked >= 1:
        findings_json["structural_constraints"].append({
            "type": "count",
            "details": {
                "target": "intermediate_with_methylsulfonyl",
                "operator": ">=",
                "value": 1
            }
        })

    # Return True if the methylsulfonyl group is in the target and at least one intermediate
    return result, findings_json
