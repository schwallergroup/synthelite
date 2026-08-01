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


HETEROCYCLES = [
    "furan", "pyran", "pyrrole", "pyridine", "pyrazole", "imidazole",
    "oxazole", "thiazole", "pyrimidine", "pyrazine", "triazole", "tetrazole",
    "indole", "benzoxazole", "benzothiazole", "benzimidazole", 
    "pyrimidinone",
    "3,4-dihydro-1H-pyrimidin-2-one"
]

HETEROCYCLE_FORMATION_REACTIONS = [
    "benzimidazole_derivatives_carboxylic-acid/ester", "benzimidazole_derivatives_aldehyde",
    "benzothiazole", "benzoxazole_arom-aldehyde", "benzoxazole_carboxylic-acid",
    "thiazole", "tetrazole_terminal", "Huisgen_Cu-catalyzed_1,4-subst",
    "1,2,4-triazole_acetohydrazide", "1,2,4-triazole_carboxylic-acid/ester",
    "pyrazole", "Paal-Knorr pyrrole", "Fischer indole", "benzofuran",
    "benzothiophene", "indole", "oxadiazole", "Formation of NOS Heterocycles"
]

SNAR_REACTION_TYPES = [
    "heteroaromatic_nuc_sub", "nucl_sub_aromatic_ortho_nitro",
    "nucl_sub_aromatic_para_nitro", "Buchwald-Hartwig", "N-arylation",
    "Ullmann-Goldberg Substitution amine", "Ullmann-Goldberg Substitution thiol",
    "Ullmann-Goldberg Substitution aryl alcohol", "Goldberg coupling",
    "Ullmann condensation", "N-arylation_heterocycles",
    "Williamson Ether Synthesis", "Williamson ether"
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects a multi-step 'core elaboration' strategy. This is defined as the
    formation of a heterocycle (from HETEROCYCLE_FORMATION_REACTIONS or de novo
    from HETEROCYCLES), which is subsequently functionalized using one or more
    SNAr or cross-coupling reactions (from SNAR_REACTION_TYPES). The strategy is
    also flagged if a heterocycle is formed and at least two such elaboration
    reactions occur anywhere in the synthesis.
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

    # Track if we found heterocycle formation and SNAr reactions
    heterocycle_formation_found = False
    heterocycle_formation_depth = float("inf")
    snar_reactions = []  # Store tuples of (depth, reaction_smiles)
    late_stage_sulfonyl = False

    def dfs_traverse(node, depth=0):
        nonlocal heterocycle_formation_found, heterocycle_formation_depth, snar_reactions, late_stage_sulfonyl, findings_json

        if node["type"] == "reaction":
            try:
                # Extract reactants and product
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                reactants_smiles = rsmi.split(">")[0].split(".")
                product_smiles = rsmi.split(">")[-1]

                # Check for heterocycle formation using specific reactions
                for rxn_type in HETEROCYCLE_FORMATION_REACTIONS:
                    if checker.check_reaction(rxn_type, rsmi):
                        print(
                            f"Heterocycle formation reaction {rxn_type} detected at depth {depth}"
                        )
                        heterocycle_formation_found = True
                        heterocycle_formation_depth = min(heterocycle_formation_depth, depth)
                        if rxn_type not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                        break

                # If no specific reaction found, check for heterocycle presence
                if not heterocycle_formation_found:
                    # Check if product contains a heterocycle
                    product_has_heterocycle = False
                    reactants_have_heterocycle = False

                    # Check which heterocycles are in the product
                    product_heterocycles = []
                    for heterocycle in HETEROCYCLES:
                        if checker.check_ring(heterocycle, product_smiles):
                            product_has_heterocycle = True
                            product_heterocycles.append(heterocycle)
                            if heterocycle not in findings_json["atomic_checks"]["ring_systems"]:
                                findings_json["atomic_checks"]["ring_systems"].append(heterocycle)
                            print(f"Product contains {heterocycle} at depth {depth}")

                    # Check if any reactant contains the same heterocycles
                    reactant_heterocycles = set()
                    for reactant in reactants_smiles:
                        if not reactant:  # Skip empty reactants
                            continue
                        for heterocycle in HETEROCYCLES:
                            if checker.check_ring(heterocycle, reactant):
                                reactants_have_heterocycle = True
                                reactant_heterocycles.add(heterocycle)
                                if heterocycle not in findings_json["atomic_checks"]["ring_systems"]:
                                    findings_json["atomic_checks"]["ring_systems"].append(heterocycle)
                                print(f"Reactant contains {heterocycle} at depth {depth}")

                    # If product has heterocycle but reactants don't have all the same heterocycles,
                    # it's a heterocycle formation
                    new_heterocycles = [
                        h for h in product_heterocycles if h not in reactant_heterocycles
                    ]
                    if product_has_heterocycle and new_heterocycles:
                        print(
                            f"Heterocycle formation detected at depth {depth} (new: {new_heterocycles})"
                        )
                        heterocycle_formation_found = True
                        heterocycle_formation_depth = min(heterocycle_formation_depth, depth)
                        if "ring_formation" not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append("ring_formation")

                # Check for SNAr reaction using the checker function
                for rxn_type in SNAR_REACTION_TYPES:
                    if checker.check_reaction(rxn_type, rsmi):
                        print(f"SNAr reaction {rxn_type} detected at depth {depth}")
                        snar_reactions.append((depth, rsmi))
                        if rxn_type not in findings_json["atomic_checks"]["named_reactions"]:
                            findings_json["atomic_checks"]["named_reactions"].append(rxn_type)
                        break

            except Exception as e:
                print(f"Error processing reaction at depth {depth}: {e}")

        # Determine the next depth based on the current node's type
        next_depth = depth
        if node["type"] != "reaction": # If current node is chemical, depth increases
            next_depth = depth + 1

        # Traverse children
        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    # Start traversal from the root
    dfs_traverse(route)

    # Check if SNAr reactions occur after heterocycle formation
    sequential_snar = False
    if heterocycle_formation_found and snar_reactions:
        # Sort SNAr reactions by depth
        snar_reactions.sort(key=lambda x: x[0])
        # Check if at least one SNAr reaction occurs after heterocycle formation
        sequential_snar = any(depth < heterocycle_formation_depth for depth, _ in snar_reactions)
        print(
            f"Sequential SNAr check: heterocycle at depth {heterocycle_formation_depth}, SNAr at depths {[d for d, _ in snar_reactions]}"
        )

    # Return True if the strategy is detected
    # Modified to ensure SNAr reactions occur after heterocycle formation
    strategy_detected = heterocycle_formation_found and (
        sequential_snar or late_stage_sulfonyl or len(snar_reactions) >= 2
    )
    print(
        f"Strategy detected: {strategy_detected} (Heterocycle: {heterocycle_formation_found} at depth {heterocycle_formation_depth}, SNAr: {len(snar_reactions)}, Sequential: {sequential_snar}, Late-stage sulfonyl: {late_stage_sulfonyl})"
    )

    # Populate structural constraints based on detected conditions
    if heterocycle_formation_found and len(snar_reactions) > 0:
        # Co-occurrence constraint
        findings_json["structural_constraints"].append({
            "type": "co-occurrence",
            "details": {
                "targets": [
                    "heterocycle_formation_event",
                    "snar_reaction_event"
                ],
                "notes": "The overall strategy requires at least one heterocycle formation event and at least one SNAr reaction event. The relationship between them is further constrained by a sequence OR a count requirement."
            }
        })

    if sequential_snar:
        # Sequence constraint
        findings_json["structural_constraints"].append({
            "type": "sequence",
            "details": {
                "before": "heterocycle_formation_event",
                "after": "snar_reaction_event",
                "notes": "This is one of two valid conditions for the elaboration step: at least one SNAr reaction must occur after the heterocycle is formed (i.e., at a lower depth)."
            }
        })

    if len(snar_reactions) >= 2:
        # Count constraint
        findings_json["structural_constraints"].append({
            "type": "count",
            "details": {
                "target": "snar_reaction_event",
                "operator": ">=",
                "value": 2,
                "notes": "This is the second of two valid conditions for the elaboration step: at least two SNAr reactions must occur anywhere in the route."
            }
        })

    return strategy_detected, findings_json
