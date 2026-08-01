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


AMIDE_FORMATION_REACTIONS = [
    "Acylation of Nitrogen Nucleophiles by Carboxylic Acids",
    "Carboxylic acid with primary amine to amide",
    "Ester with primary amine to amide",
    "Ester with ammonia to amide",
    "Ester with secondary amine to amide",
    "Acyl chloride with primary amine to amide (Schotten-Baumann)",
    "Acyl chloride with secondary amine to amide",
    "Acyl chloride with ammonia to amide",
    "Schotten-Baumann_amide",
]

COUPLING_REACTIONS = [
    "Suzuki coupling with boronic acids",
    "Suzuki coupling with boronic esters",
    "Suzuki coupling with boronic acids OTf",
    "Suzuki coupling with boronic esters OTf",
    "Suzuki",
    "Negishi coupling",
    "Negishi",
    "Stille reaction_aryl",
    "Stille reaction_vinyl",
    "Stille",
    "Heck terminal vinyl",
    "Heck_terminal_vinyl",
    "Sonogashira alkyne_aryl halide",
    "Sonogashira acetylene_aryl halide",
    "Hiyama-Denmark Coupling",
    "Kumada cross-coupling",
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects a specific synthetic pattern where an amide is formed in an earlier step (higher depth) than a subsequent cross-coupling reaction. The detection of both reaction types relies on checking against predefined lists of named reactions.
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

    # Track transformations and their depths
    amide_formation = False
    coupling_reaction = False

    amide_depth = -1
    coupling_depth = -1

    def dfs_traverse(node, depth=0):
        nonlocal amide_formation, coupling_reaction
        nonlocal amide_depth, coupling_depth, findings_json

        if node["type"] == "reaction":
            if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
                rsmi = node["metadata"]["mapped_reaction_smiles"]

                # Check for amide formation using predefined reaction types
                for rxn_type in AMIDE_FORMATION_REACTIONS:
                    if checker.check_reaction(rxn_type, rsmi):
                        amide_formation = True
                        amide_depth = depth
                        if rxn_type not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                        break

                # Check for coupling reaction using predefined reaction types
                for rxn_type in COUPLING_REACTIONS:
                    if checker.check_reaction(rxn_type, rsmi):
                        coupling_reaction = True
                        coupling_depth = depth
                        if rxn_type not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                        break

        # Traverse children
        for child in node.get("children", []):
            # New logic for depth calculation
            new_depth = depth
            if node["type"] != "reaction":  # If current node is not a reaction (e.g., chemical)
                new_depth = depth + 1
            dfs_traverse(child, new_depth)

    # Start traversal
    dfs_traverse(route)

    # Check if amide formation occurs before coupling (higher depth = earlier in synthesis)
    amide_before_coupling = amide_formation and coupling_reaction and amide_depth > coupling_depth

    if amide_before_coupling:
        # Add the structural constraint if the condition is met
        findings_json["structural_constraints"].append({
            "type": "sequence",
            "details": {
                "before": {
                    "group_name": "amide_formation",
                    "any_of": AMIDE_FORMATION_REACTIONS
                },
                "after": {
                    "group_name": "cross_coupling",
                    "any_of": COUPLING_REACTIONS
                }
            }
        })

    return amide_before_coupling, findings_json
