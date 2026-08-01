#!/bin/python

"""LM-defined function for strategy description."""

from rdkit.Chem import AllChem, rdFMCS
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


def main(route):
    """
    This function detects a synthetic strategy involving late-stage introduction
    of a trifluoromethyl group.
    """
    has_late_fg_introduction = False

    def dfs_traverse(node, current_depth=0):
        nonlocal has_late_fg_introduction

        if node["type"] == "reaction":
            if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
                # Use the recursively passed depth. depth=1 is the final reaction.
                depth = current_depth

                # Consider reactions at depth 1 or 2 as late-stage.
                if depth in [1, 2]:
                    rsmi = node["metadata"]["mapped_reaction_smiles"]
                    reactants_smiles = rsmi.split(">")[0].split(".")
                    product_smiles = rsmi.split(">")[-1]

                    # Check for trifluoromethyl group in product
                    product_has_fg = checker.check_fg("Methyl sulfonyl", product_smiles)

                    if product_has_fg:
                        # Check if any reactant doesn't have CF3 (indicating introduction)
                        reactants_with_fg = [
                            checker.check_fg("Methyl sulfonyl", reactant)
                            for reactant in reactants_smiles
                        ]

                        # Case 1: CF3 group is transferred from a reagent.
                        if not all(reactants_with_fg) and any(reactants_with_fg):
                            has_late_fg_introduction = True

                        # Case 2: CF3 group is formed de novo.
                        if not any(reactants_with_fg):
                            has_late_fg_introduction = True

        # Traverse children with incremented depth
        for child in node.get("children", []):
            # New logic: depth increases only from chemical to reaction node
            if node["type"] == "reaction":
                dfs_traverse(child, current_depth)
            else:  # Assuming 'chemical' or other types
                dfs_traverse(child, current_depth + 1)

    # Start traversal from the root
    dfs_traverse(route)

    return has_late_fg_introduction
