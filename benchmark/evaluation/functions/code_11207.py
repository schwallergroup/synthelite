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


# Refactored lists as module-level constants
HETEROCYCLES_OF_INTEREST = [
    "indole", "benzimidazole", "benzoxazole", "benzothiazole", "quinoline",
    "isoquinoline", "pyridine", "pyrimidine", "pyrazole", "imidazole",
    "oxazole", "thiazole", "furan", "thiophene", "pyrrole", "triazole", "tetrazole",
]

SUZUKI_REACTION_TYPES = [
    "Suzuki coupling with boronic acids", "Suzuki coupling with boronic esters",
    "Suzuki coupling with boronic acids OTf", "Suzuki coupling with boronic esters OTf",
    "Suzuki coupling with sulfonic esters", "{Suzuki}",
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects if the synthesis uses a late-stage Suzuki coupling to connect
    a heterocycle (like indole) to a pre-formed core structure.
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

    late_stage_coupling_found = False

    def dfs_traverse(node, depth=0):
        nonlocal late_stage_coupling_found, findings_json

        if node["type"] == "reaction" and depth <= 2:
            if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
                try:
                    rsmi = node["metadata"]["mapped_reaction_smiles"]
                    reactants_part = rsmi.split(">")[0]
                    reactants = reactants_part.split(".")
                    product = rsmi.split(">")[-1]

                    # Check if this is a Suzuki coupling
                    is_suzuki = False
                    for reaction_type in SUZUKI_REACTION_TYPES:
                        if checker.check_reaction(reaction_type, rsmi):
                            is_suzuki = True
                            if reaction_type not in findings_json["atomic_checks"]["named_reactions"]:
                                findings_json["atomic_checks"]["named_reactions"].append(reaction_type)
                            break

                    if is_suzuki:
                        # Check if product contains a heterocycle
                        heterocycle_in_product = False
                        heterocycle_name = None

                        for heterocycle in HETEROCYCLES_OF_INTEREST:
                            if checker.check_ring(heterocycle, product):
                                heterocycle_in_product = True
                                heterocycle_name = heterocycle
                                if heterocycle not in findings_json["atomic_checks"]["ring_systems"]:
                                    findings_json["atomic_checks"]["ring_systems"].append(heterocycle)
                                break

                        if heterocycle_in_product:
                            # Check which reactant(s) contain the heterocycle
                            heterocycle_reactant_idx = None
                            other_reactant_idx = None

                            for i, reactant in enumerate(reactants):
                                if checker.check_ring(heterocycle_name, reactant):
                                    heterocycle_reactant_idx = i
                                else:
                                    other_reactant_idx = i

                            # If we found both a reactant with the heterocycle and one without,
                            # and the product has the heterocycle, this is likely a coupling
                            if (
                                heterocycle_reactant_idx is not None
                                and other_reactant_idx is not None
                            ):
                                late_stage_coupling_found = True
                                # Add structural constraints if the main condition is met
                                if {"type": "co-occurrence", "details": {"targets": ["Suzuki coupling", "heterocycle_from_list"], "scope": "within_reaction"}} not in findings_json["structural_constraints"]:
                                    findings_json["structural_constraints"].append({"type": "co-occurrence", "details": {"targets": ["Suzuki coupling", "heterocycle_from_list"], "scope": "within_reaction"}})
                                if {"type": "positional", "details": {"target": "Suzuki coupling", "position": "depth <= 2"}} not in findings_json["structural_constraints"]:
                                    findings_json["structural_constraints"].append({"type": "positional", "details": {"target": "Suzuki coupling", "position": "depth <= 2"}})

                except Exception:
                    # Silently handle parsing errors or other issues
                    pass

        for child in node.get("children", []):
            # New logic for depth calculation
            new_depth = depth
            if node["type"] != "reaction": # If current node is chemical, depth increases
                new_depth = depth + 1
            # If current node is reaction, depth remains the same
            dfs_traverse(child, new_depth)

    dfs_traverse(route)
    print(f"Final result: {late_stage_coupling_found}")
    return late_stage_coupling_found, findings_json
