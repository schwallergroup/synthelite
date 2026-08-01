from typing import Tuple, Dict
import copy

from rdkit import Chem

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

ALDOL_REACTIONS = ["Aldol condensation"]

# An aldol/Knoevenagel condensation forms an alpha,beta-unsaturated carbonyl
# (enone/enal): an aliphatic C=C conjugated to a ketone or aldehyde carbonyl.
# The condensation is often embedded in a multicomponent reaction whose full
# product the 2-component SMIRKS above cannot match, so we also detect it
# structurally: an enone present in the product but absent from the reactants.
ENONE_PATTERNS = [
    Chem.MolFromSmarts("[C]=[C][CX3](=[OX1])[#6]"),   # alpha,beta-unsaturated ketone
    Chem.MolFromSmarts("[C]=[C][CX3H1]=[OX1]"),       # alpha,beta-unsaturated aldehyde
]


def _has_enone(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(patt) for patt in ENONE_PATTERNS if patt is not None)


def main(route) -> Tuple[bool, Dict]:
    """
    Detects late-stage aldol (or Knoevenagel) condensations and records the
    depth at which each occurs. Depth follows the repo convention (see
    code_3597): it increments on chemical (non-reaction) nodes, so small depths
    are late-stage (near the target product) and large depths are early-stage
    (near the starting materials).

    A reaction counts as an aldol condensation if either the ``Aldol
    condensation`` reaction template matches, or an alpha,beta-unsaturated
    carbonyl (enone/enal) is present in the product but not in any reactant
    (i.e. it was formed in this step). The latter, structural signal captures
    aldol condensations embedded in multicomponent reactions that the strict
    2-component template cannot match.

    Returns a ``(found, findings)`` tuple where ``findings['at_depths']`` holds
    the depth of every detected condensation, so callers can decide whether it
    is late-stage.
    """
    findings_template = {
        "atomic_checks": {
            "named_reactions": [],
            "ring_systems": [],
            "functional_groups": []
        },
        "structural_constraints": [],
        "at_depths": []
    }
    findings_json = copy.deepcopy(findings_template)

    found_aldol = False

    def dfs_traverse(node, depth=0):
        nonlocal found_aldol, findings_json

        if node["type"] == "reaction":
            try:
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                reactants_part = rsmi.split(">")[0]
                product_part = rsmi.split(">")[-1]

                matched = False

                # 1) Explicit named aldol-condensation template.
                for reaction_type in ALDOL_REACTIONS:
                    if checker.check_reaction(reaction_type, rsmi):
                        findings_json["atomic_checks"]["named_reactions"].append(reaction_type)
                        matched = True
                        break

                # 2) Structural: an enone formed in this step (product has one,
                #    no reactant does).
                if not matched:
                    product_has_enone = any(
                        _has_enone(p) for p in product_part.split(".")
                    )
                    reactant_has_enone = any(
                        _has_enone(r) for r in reactants_part.split(".")
                    )
                    if product_has_enone and not reactant_has_enone:
                        findings_json["atomic_checks"]["functional_groups"].append(
                            "enone_formation"
                        )
                        matched = True

                if matched:
                    found_aldol = True
                    findings_json["at_depths"].append(depth)

            except Exception as e:
                print(f"Error processing reaction node: {e}")

        # Depth increments on chemical (non-reaction) nodes, not reaction nodes.
        next_depth = depth
        if node["type"] != "reaction":
            next_depth = depth + 1

        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    dfs_traverse(route)

    return found_aldol, findings_json
