from typing import Tuple, Dict, List
import copy
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


HALOGENATION_REACTIONS_OF_INTEREST = [
    "Aromatic fluorination",
    "Aromatic chlorination",
    "Aromatic bromination",
    "Aromatic iodination",
    "Chlorination",
    "Fluorination",
    "Iodination",
    "Bromination",
]

def main(route) -> Tuple[bool, Dict]:
    """
    Checks for late-stage halogenation at the final synthetic step. This is identified by checking if the reaction is one of the following types: Aromatic fluorination, Aromatic chlorination, Aromatic bromination, Aromatic iodination, Chlorination, Fluorination, Iodination, or Bromination.
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

    # Track if we found late-stage halogenation
    found_late_stage_halogenation = False

    def dfs_traverse(node, depth=0):
        nonlocal found_late_stage_halogenation, findings_json

        if node["type"] == "reaction" and depth <= 1:  # Final reaction step
            if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
                rsmi = node["metadata"]["mapped_reaction_smiles"]

                # Check if this is a halogenation reaction
                for reaction_type in HALOGENATION_REACTIONS_OF_INTEREST:
                    if checker.check_reaction(reaction_type, rsmi):
                        found_late_stage_halogenation = True
                        findings_json["atomic_checks"]["named_reactions"].append(reaction_type)
                        # Add the structural constraint if this is the final step and a halogenation reaction is found
                        findings_json["structural_constraints"].append({
                            "type": "positional",
                            "details": {
                                "targets": [
                                    "Aromatic fluorination",
                                    "Aromatic chlorination",
                                    "Aromatic bromination",
                                    "Aromatic iodination",
                                    "Chlorination",
                                    "Fluorination",
                                    "Iodination",
                                    "Bromination"
                                ],
                                "position": "last_stage",
                                "match_logic": "any"
                            }
                        })
                        return

        # Traverse children with incremented depth
        for child in node.get("children", []):
            # New logic: depth increases only when traversing from chemical to reaction
            # Depth remains the same when traversing from reaction to chemical
            if node["type"] == "reaction":
                dfs_traverse(child, depth)
            else: # node['type'] == 'chemical'
                dfs_traverse(child, depth + 1)

    # Start traversal
    dfs_traverse(route)

    return found_late_stage_halogenation, findings_json
