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


CARBONYL_FUNCTIONALIZATION_RXNS = [
    "Reduction of aldehydes and ketones to alcohols",
    "Reductive amination with aldehyde",
    "Reductive secondary amination with aldehyde",
    "Reductive amination with ketone",
    "Reductive secondary amination with ketone",
    "Addition of primary amines to aldehydes/thiocarbonyls",
    "Addition of primary amines to ketones/thiocarbonyls",
    "Addition of secondary amines to aldehydes/thiocarbonyls",
    "Addition of secondary amines to ketones/thiocarbonyls",
    # "Wittig reaction with triphenylphosphorane",
    # "Wittig",
    # "Aldol condensation",
    # "Grignard from aldehyde to alcohol",
    # "Grignard from ketone to alcohol",
    # "Olefination of ketones with Grignard reagents",
    # "Olefination of aldehydes with Grignard reagents",
    # "Ketone/aldehyde to hydrazone",
    # "Nef reaction (nitro to ketone)",
    # "Oxidation of alkene to aldehyde",
]

CARBONYL_FUNCTIONALIZATION_PRODUCTS = [
    "Primary alcohol",
    "Secondary alcohol",
    "Tertiary alcohol",
    "Primary amine",
    "Secondary amine",
    "Tertiary amine",
    "Substituted imine",
    "Unsubstituted imine",
    "Hydrazone",
    "Oxime",
    "Alkene",
    "Alkyne",
    "Azide",
    "Nitrile",
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects late-stage (final 3 steps) functionalization of a carbonyl group (ketone or aldehyde).
    A reaction is flagged if it matches a known carbonyl transformation from `CARBONYL_FUNCTIONALIZATION_RXNS`,
    or if a reactant carbonyl is consumed to form a product functional group listed in `CARBONYL_FUNCTIONALIZATION_PRODUCTS`.
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

    # Track synthesis characteristics
    is_linear = True  # Assume linear until proven otherwise
    has_late_carbonyl_functionalization = False

    def dfs_traverse(node, depth=0):
        nonlocal is_linear, has_late_carbonyl_functionalization, findings_json

        if node["type"] == "reaction":
            # Extract reactants and product
            try:
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                reactants_smiles = rsmi.split(">")[0].split(".")
                product_smiles = rsmi.split(">")[-1]

                # Check for late-stage carbonyl functionalization
                if depth <= 3:
                    # Check if any reactant has a carbonyl group
                    has_carbonyl_reactant = False
                    for reactant_smiles in reactants_smiles:
                        if reactant_smiles:
                            if checker.check_fg("Ketone", reactant_smiles):
                                has_carbonyl_reactant = True
                                if "Ketone" not in findings_json["atomic_checks"]["functional_groups"]:
                                    findings_json["atomic_checks"]["functional_groups"].append("Ketone")
                                break
                            if checker.check_fg("Aldehyde", reactant_smiles):
                                has_carbonyl_reactant = True
                                if "Aldehyde" not in findings_json["atomic_checks"]["functional_groups"]:
                                    findings_json["atomic_checks"]["functional_groups"].append("Aldehyde")
                                break

                    # Check if the reaction is a known carbonyl functionalization
                    is_carbonyl_rxn = False
                    if has_carbonyl_reactant:
                        for rxn_type in CARBONYL_FUNCTIONALIZATION_RXNS:
                            if checker.check_reaction(rxn_type, rsmi):
                                is_carbonyl_rxn = True
                                if rxn_type not in findings_json["atomic_checks"]["named_reactions"]:
                                    findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                                break

                    # If we couldn't identify a specific reaction type, check for functional group changes
                    if not is_carbonyl_rxn and has_carbonyl_reactant:
                        # Check if carbonyl is converted to another functional group
                        has_carbonyl_product = checker.check_fg(
                            "Ketone", product_smiles
                        ) or checker.check_fg("Aldehyde", product_smiles)

                        if not has_carbonyl_product:
                            # Check for common carbonyl functionalization products
                            for fg in CARBONYL_FUNCTIONALIZATION_PRODUCTS:
                                if checker.check_fg(fg, product_smiles):
                                    is_carbonyl_rxn = True
                                    if fg not in findings_json["atomic_checks"]["functional_groups"]:
                                        findings_json["atomic_checks"]["functional_groups"].append(fg)
                                    break

                    if is_carbonyl_rxn:
                        has_late_carbonyl_functionalization = True
                        # Add the structural constraint if detected
                        # This corresponds to the 'positional' constraint in the strategy JSON
                        if {"type": "positional", "details": {"target": "carbonyl_functionalization", "position": "late_stage", "max_depth": 3}} not in findings_json["structural_constraints"]:
                            findings_json["structural_constraints"].append({
                                "type": "positional",
                                "details": {
                                    "target": "carbonyl_functionalization",
                                    "position": "late_stage",
                                    "max_depth": 3
                                }
                            })

            except Exception as e:
                print(f"Error processing reaction at depth {depth}: {e}")

        # Process children
        for child in node.get("children", []):
            # Determine the new depth based on the current node's type
            new_depth = depth
            if node["type"] != "reaction": # If current node is chemical, depth increases
                new_depth = depth + 1
            # If current node is reaction, depth remains the same
            
            dfs_traverse(child, new_depth)

    # Start traversal
    dfs_traverse(route)

    result = is_linear and has_late_carbonyl_functionalization
    print(
        f"Strategy detection result: {result} (Linear: {is_linear}, Late carbonyl functionalization: {has_late_carbonyl_functionalization})"
    )
    return result, findings_json
