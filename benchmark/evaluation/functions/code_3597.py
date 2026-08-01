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


HETEROCYCLES_OF_INTEREST = [
    "pyridine", "pyrrole", "furan", "thiophene", "imidazole", "oxazole",
    "thiazole", "pyrazole", "isoxazole", "isothiazole", "pyrimidine",
    "pyrazine", "pyridazine", "triazole", "tetrazole", "indole",
    "benzofuran", "benzothiophene", "benzimidazole", "benzoxazole",
    "benzothiazole", "quinoline", "isoquinoline", "piperidine",
    "tetrahydrofuran", "tetrahydropyran", "morpholine", "thiomorpholine",
    "oxoisoindolinone",
    "piperidine-2,6-dione",
    "pyrimidinone",
    "3,4-dihydro-1H-pyrimidin-2-one",
    "piperazine"
]

HETEROCYCLE_FORMING_REACTIONS = [
    "Formation of NOS Heterocycles", "Paal-Knorr pyrrole synthesis",
    "benzimidazole_derivatives_carboxylic-acid/ester",
    "benzimidazole_derivatives_aldehyde", "benzothiazole",
    "benzoxazole_arom-aldehyde", "benzoxazole_carboxylic-acid", "thiazole",
    "tetrazole_terminal", "tetrazole_connect_regioisomere_1",
    "tetrazole_connect_regioisomere_2", "1,2,4-triazole_acetohydrazide",
    "1,2,4-triazole_carboxylic-acid/ester", "3-nitrile-pyridine", "pyrazole",
    "Fischer indole", "Friedlaender chinoline", "benzofuran",
    "benzothiophene", "indole", "oxadiazole", "imidazole",
    "Huisgen_Cu-catalyzed_1,4-subst", "Huisgen_Ru-catalyzed_1,5_subst",
    "imidazole_from_carbonyldiimidazole"
    # "oxoisoindolinone"
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects heterocycle formation by checking for specific named reactions or by
    identifying the de novo formation of a heterocycle. The specific named
    reactions are enumerated in the `HETEROCYCLE_FORMING_REACTIONS` list, and
    the heterocycles checked for de novo formation are enumerated in the
    `HETEROCYCLES_OF_INTEREST` list.
    """
    findings_template = {
        "atomic_checks": {
            "named_reactions": [],
            "ring_systems": [],
            "functional_groups": []
        },
        "structural_constraints": [],
        'at_depths': []
    }
    findings_json = copy.deepcopy(findings_template)

    found_heterocycle_formation = False

    def dfs_traverse(node, depth=0):
        nonlocal found_heterocycle_formation, findings_json

        # if found_heterocycle_formation:
        #     return  # Early exit if we already found what we're looking for

        if node["type"] == "reaction":
            try:
                # Extract reactants and product
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                reactants_part = rsmi.split(">")[0]
                product_part = rsmi.split(">")[-1]

                reactants = reactants_part.split(".")
                product = product_part

                # Check if this is a known heterocycle-forming reaction
                for reaction_type in HETEROCYCLE_FORMING_REACTIONS:
                    if checker.check_reaction(reaction_type, rsmi):
                        print(f"Found heterocycle-forming reaction: {reaction_type}")
                        found_heterocycle_formation = True
                        findings_json["atomic_checks"]["named_reactions"].append(reaction_type)
                        findings_json['at_depths'].append(depth)
                        return

                # Check for heterocycle formation by structure comparison
                heterocycle_in_product = False
                # heterocycle_in_reactants = False # This variable is not used after its assignment

                # Check which heterocycles are in the product
                product_heterocycles = []
                for heterocycle in HETEROCYCLES_OF_INTEREST:
                    if checker.check_ring(heterocycle, product):
                        product_heterocycles.append(heterocycle)
                        heterocycle_in_product = True

                # If no heterocycles in product, no need to check reactants
                if not heterocycle_in_product:
                    pass
                else:
                    # Check which heterocycles are in the reactants
                    reactant_heterocycles = set()
                    for reactant in reactants:
                        for heterocycle in HETEROCYCLES_OF_INTEREST:
                            if checker.check_ring(heterocycle, reactant):
                                reactant_heterocycles.add(heterocycle)
                                # heterocycle_in_reactants = True # This variable is not used after its assignment

                    # Check if any heterocycle in product is not in reactants
                    new_heterocycles = [
                        h for h in product_heterocycles if h not in reactant_heterocycles
                    ]

                    if new_heterocycles:
                        print(f"Found heterocycle formation: {new_heterocycles}")
                        found_heterocycle_formation = True
                        findings_json["atomic_checks"]["ring_systems"].extend(new_heterocycles)
                        findings_json['at_depths'].append(depth)
                        # Add a generic 'ring_formation' to named_reactions if a new heterocycle is formed
                        if "ring_formation" not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append("ring_formation")

            except Exception as e:
                print(f"Error processing reaction node: {e}")

        # Determine the next depth based on the current node's type
        next_depth = depth
        if node["type"] != "reaction": # This means it's a chemical node
            next_depth = depth + 1

        # Continue traversing the tree
        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    # Start traversal from the root
    dfs_traverse(route)

    print(f"Heterocycle formation strategy detection:")
    print(f"  Heterocycle formation found: {found_heterocycle_formation}")

    return found_heterocycle_formation, findings_json
