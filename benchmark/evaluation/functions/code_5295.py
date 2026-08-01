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


# Refactoring for Enumeration: Isolate the list of rings
CYCLIZATION_RING_SYSTEMS = [
    "pyrrole", "pyridine", "pyrazole", "imidazole", "oxazole", "thiazole",
    "furan", "thiophene", "pyran", "benzoxazole", "benzothiazole",
    "benzimidazole", "indole", "quinoline", "isoquinoline", "oxirane",
    "aziridine", "azetidine", "pyrrolidine", "piperidine", "piperazine",
    "morpholine", "thiomorpholine", "tetrahydrofuran", "tetrahydropyran",
    "dioxane", "dioxolane", "oxetane", "thietane", "thiolane", "thiane",
    "oxazolidine", "thiazolidine", "isoxazole", "isothiazole", "oxadiazole",
    "thiadiazole", "triazole", "tetrazole", "pyrimidine", "pyrazine", "pyridazine"
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects a late-stage (final or penultimate step) formation of a specific ring system via intramolecular cyclization.
    The strategy is confirmed by verifying that a ring from the CYCLIZATION_RING_SYSTEMS list is formed in the reaction
    (i.e., present in the product but absent from all reactants).
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

    found_late_cyclization = False

    def dfs_traverse(node, depth=0):
        nonlocal found_late_cyclization, findings_json

        if node["type"] == "reaction" and depth <= 2:  # Focus on late-stage reactions
            try:
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                reactants_smiles = rsmi.split(">")[0].split(".")
                product_smiles = rsmi.split(">")[-1]

                # Check if any ring type is in product but not in all reactants
                new_ring_formed = False
                for ring in CYCLIZATION_RING_SYSTEMS:
                    # Check if ring is in product
                    if checker.check_ring(ring, product_smiles):
                        # Check if ring is not in any reactant
                        ring_in_reactants = any(
                            checker.check_ring(ring, r_smi) for r_smi in reactants_smiles
                        )
                        if not ring_in_reactants:
                            new_ring_formed = True
                            findings_json["atomic_checks"]["ring_systems"].append(ring)
                            break

                if new_ring_formed:
                    found_late_cyclization = True
                    # Add the structural constraint if a late-stage cyclization is found
                    findings_json["structural_constraints"].append({
                        "type": "positional",
                        "details": {
                            "target": "ring_formation",
                            "position": "late_stage (depth <= 2)"
                        }
                    })

            except Exception as e:
                # For this function's purpose, we can ignore reactions that fail to parse.
                pass

        # Traverse children
        for child in node.get("children", []):
            # New logic for depth calculation
            new_depth = depth
            if node["type"] != "reaction": # If current node is chemical, depth increases
                new_depth = depth + 1
            # If current node is reaction, depth remains the same
            dfs_traverse(child, new_depth)

    # Start traversal
    dfs_traverse(route)

    return found_late_cyclization, findings_json
