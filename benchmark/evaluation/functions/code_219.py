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


HETEROCYCLIC_PARTNERS = [
    "pyridine",
    "pyrrole",
    "furan",
    "thiophene",
    "imidazole",
    "oxazole",
    "thiazole",
    "pyrazole",
    "triazole",
    "tetrazole",
    "pyrimidine",
    "pyrazine",
    "indole",
    "benzimidazole",
    "benzoxazole",
    "benzothiazole",
    "quinoline",
    "isoquinoline",
    "piperidine",
    "piperazine",
    "morpholine",
    "thiomorpholine",
]

def main(route) -> Tuple[bool, Dict]:
    """
    Detects a late-stage (depth <= 2) amide coupling reaction where at least one reactant is a heterocycle from the HETEROCYCLIC_PARTNERS list.
    """
    print("Starting late_stage_amide_coupling_strategy analysis")
    late_stage_amide_coupling = False

    findings_template = {
        "atomic_checks": {
            "named_reactions": [],
            "ring_systems": [],
            "functional_groups": []
        },
        "structural_constraints": []
    }
    findings_json = copy.deepcopy(findings_template)

    def dfs_traverse(node, depth=0):
        nonlocal late_stage_amide_coupling, findings_json

        # Check only late-stage reactions (depth 0, 1, or 2)
        if (
            node["type"] == "reaction"
            and depth <= 2
            and "metadata" in node
            and "mapped_reaction_smiles" in node["metadata"]
        ):
            rsmi = node["metadata"]["mapped_reaction_smiles"]
            print(f"Examining reaction at depth {depth}: {rsmi}")
            reactants = rsmi.split(">")[0].split(".")
            product = rsmi.split(">")[-1]

            is_amide_coupling = False
            detected_reaction_names = []

            # Check if this is an amide coupling reaction using the checker
            amide_coupling_reactions = [
                "Acylation of Nitrogen Nucleophiles by Carboxylic Acids",
                "Acylation of Nitrogen Nucleophiles by Acyl/Thioacyl/Carbamoyl Halides and Analogs_N",
                "Carboxylic acid with primary amine to amide",
                # "Primary amide with primary amine to amide",
                "Amide exchange",
                "Acyl chloride with primary amine to amide (Schotten-Baumann)",
                "Acyl halide with primary amine to amide",
                "Acyl chloride with secondary amine to amide",
                "Acyl halide with secondary amine to amide",
                "Ester with primary amine to amide",
                "Ester with secondary amine to amide",
                "Schotten-Baumann_amide",
                "Acylation of primary amines",
                "Acylation of secondary amines",
            ]

            for r_name in amide_coupling_reactions:
                if checker.check_reaction(r_name, rsmi):
                    is_amide_coupling = True
                    detected_reaction_names.append(r_name)
                    if r_name not in findings_json["atomic_checks"]["named_reactions"]:
                        findings_json["atomic_checks"]["named_reactions"].append(r_name)

            # If no specific reaction type is detected, check for general amide formation pattern
            if not is_amide_coupling:
                # Check if product has amide that wasn't in reactants
                has_amide_product = (
                    checker.check_fg("Primary amide", product)
                    or checker.check_fg("Secondary amide", product)
                    or checker.check_fg("Tertiary amide", product)
                )
                if checker.check_fg("Primary amide", product) and "Primary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                    findings_json["atomic_checks"]["functional_groups"].append("Primary amide")
                if checker.check_fg("Secondary amide", product) and "Secondary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                    findings_json["atomic_checks"]["functional_groups"].append("Secondary amide")
                if checker.check_fg("Tertiary amide", product) and "Tertiary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                    findings_json["atomic_checks"]["functional_groups"].append("Tertiary amide")

                has_amide_reactants = False
                for r in reactants:
                    if checker.check_fg("Primary amide", r):
                        has_amide_reactants = True
                        if "Primary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Primary amide")
                    if checker.check_fg("Secondary amide", r):
                        has_amide_reactants = True
                        if "Secondary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Secondary amide")
                    if checker.check_fg("Tertiary amide", r):
                        has_amide_reactants = True
                        if "Tertiary amide" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Tertiary amide")

                # Check for acid and amine components
                has_acid = False
                for r in reactants:
                    if checker.check_fg("Carboxylic acid", r):
                        has_acid = True
                        if "Carboxylic acid" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Carboxylic acid")

                has_amine = False
                for r in reactants:
                    if checker.check_fg("Primary amine", r):
                        has_amine = True
                        if "Primary amine" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Primary amine")
                    if checker.check_fg("Secondary amine", r):
                        has_amine = True
                        if "Secondary amine" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Secondary amine")
                    if checker.check_fg("Aniline", r):
                        has_amine = True
                        if "Aniline" not in findings_json["atomic_checks"]["functional_groups"]:
                            findings_json["atomic_checks"]["functional_groups"].append("Aniline")

                # If product has amide, reactants have acid and amine, and reactants don't have amide
                if has_amide_product and has_acid and has_amine and not has_amide_reactants:
                    print("Detected amide formation based on functional group analysis")
                    is_amide_coupling = True
                    if "amide_formation" not in findings_json["atomic_checks"]["named_reactions"]:
                        findings_json["atomic_checks"]["named_reactions"].append("amide_formation")

            if is_amide_coupling:
                # If an amide coupling is confirmed, check if any reactant contains a heterocycle.
                is_heterocyclic_partner_present = False
                detected_ring_systems = []
                for r in reactants:
                    for ring_type in HETEROCYCLIC_PARTNERS:
                        if checker.check_ring(ring_type, r):
                            is_heterocyclic_partner_present = True
                            if ring_type not in findings_json["atomic_checks"]["ring_systems"]:
                                findings_json["atomic_checks"]["ring_systems"].append(ring_type)
                            break # Found one, no need to check other rings for this reactant
                    if is_heterocyclic_partner_present: # Found for this reactant, no need to check other reactants
                        break

                if is_heterocyclic_partner_present:
                    print(
                        f"✓ Confirmed late-stage amide coupling with heterocyclic partner at depth {depth}"
                    )
                    late_stage_amide_coupling = True
                    # Add structural constraints if both conditions are met
                    # This corresponds to the "co-occurrence" constraint
                    co_occurrence_constraint = {
                        "type": "co-occurrence",
                        "details": {
                            "targets": [
                                "amide_coupling",
                                "heterocycle_in_reactant"
                            ],
                            "scope": "reaction_step"
                        }
                    }
                    if co_occurrence_constraint not in findings_json["structural_constraints"]:
                        findings_json["structural_constraints"].append(co_occurrence_constraint)

                    # This corresponds to the "positional" constraint
                    positional_constraint = {
                        "type": "positional",
                        "details": {
                            "target": "amide_coupling_with_heterocycle",
                            "position": "late_stage",
                            "condition": "<= 2"
                        }
                    }
                    if positional_constraint not in findings_json["structural_constraints"]:
                        findings_json["structural_constraints"].append(positional_constraint)

        # Traverse children with modified depth calculation
        for child in node.get("children", []):
            if node["type"] == "reaction":
                # Depth remains the same when traversing from a reaction node to a chemical node
                dfs_traverse(child, depth)
            else:
                # Depth increases when traversing from a chemical node to a reaction node
                dfs_traverse(child, depth + 1)

    # Start traversal
    dfs_traverse(route)

    print(f"Final result: late_stage_amide_coupling = {late_stage_amide_coupling}")
    return late_stage_amide_coupling, findings_json
