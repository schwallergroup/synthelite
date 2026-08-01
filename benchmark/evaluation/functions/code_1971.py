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


# Refactoring for Enumeration: Isolate the list of reactions.
CC_BOND_FORMING_REACTIONS = [
    "Suzuki",
    "Negishi",
    "Heck",
    "Stille",
    "Sonogashira",
    "Kumada",
    "Hiyama-Denmark",
    "Friedel-Crafts alkylation",
    "Diels-Alder",
    "Wittig",
    "Grignard",
    "Aldol condensation",
    "Michael addition",
    "Aryllithium cross-coupling",
    "Catellani",
    "decarboxylative_coupling",
    "A3 coupling",
]

def main(route) -> Tuple[bool, Dict]:
    """
    This function detects if a specific, named C-C bond forming reaction occurs in the late stage of synthesis (low depth).
    The reactions checked are defined in the CC_BOND_FORMING_REACTIONS list.
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

    cc_bond_formation_depths = []
    max_depth = 0

    def dfs_traverse(node, depth=0):
        nonlocal max_depth, findings_json
        max_depth = max(max_depth, depth)

        if node["type"] == "reaction" and "rsmi" in node.get("metadata", {}):
            rsmi = node["metadata"]["mapped_reaction_smiles"]

            is_cc_formation = False

            # Check against known C-C bond forming reactions
            for rxn_type in CC_BOND_FORMING_REACTIONS:
                try:
                    if checker.check_reaction(rxn_type, rsmi):
                        is_cc_formation = True
                        findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                        break
                except Exception:
                    # Silently ignore checker errors for robustness
                    pass

            if is_cc_formation:
                cc_bond_formation_depths.append(depth)

        for child in node.get("children", []):
            # New logic for depth calculation
            if node["type"] == "reaction":
                # If current node is a reaction, depth remains the same for its children (chemicals)
                dfs_traverse(child, depth)
            else:
                # If current node is a chemical, depth increases for its children (reactions)
                dfs_traverse(child, depth + 1)

    dfs_traverse(route)

    # Define late stage as the first third of the synthesis depth
    late_stage_threshold = max(1, max_depth // 3)
    result = any(depth <= late_stage_threshold for depth in cc_bond_formation_depths)

    if result:
        # Add the structural constraint if the condition is met
        findings_json["structural_constraints"].append({
            "type": "positional",
            "details": {
                "target": "any_reaction_from_group",
                "group": [
                    "Suzuki",
                    "Negishi",
                    "Heck",
                    "Stille",
                    "Sonogashira",
                    "Kumada",
                    "Hiyama-Denmark",
                    "Friedel-Crafts alkylation",
                    "Diels-Alder",
                    "Wittig",
                    "Grignard",
                    "Aldol condensation",
                    "Michael addition",
                    "Aryllithium cross-coupling",
                    "Catellani",
                    "decarboxylative_coupling",
                    "A3 coupling"
                ],
                "position_check": "depth <= max_depth / 3"
            }
        })

    return result, findings_json
