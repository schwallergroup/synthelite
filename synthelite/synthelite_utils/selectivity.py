"""Process multiple possible routes to flag potential selectivity issues."""

import re
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from functools import lru_cache


@lru_cache(maxsize=None)
def check_template(template: str, reactants: str) -> frozenset:
    """
    # input: a template, and proposed reactions (reactants obtained by retro application of template to product)
    # output: does forward template yield more than one different product?
    """

    # get forward template
    fwd_template = template.split(">>")[1] + ">>" + template.split(">>")[0]
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)

    reactants = tuple(Chem.MolFromSmiles(r) for r in reactants.split("."))

    # get all permutations of a set
    def permutations(iterable):
        from itertools import permutations

        return list(permutations(iterable))

    def process_product(prd):
        p = Chem.MolToSmiles(prd)
        p = re.sub(r"(?<=[^\*])(:\d+)]", "]", p)
        return Chem.MolToSmiles(Chem.MolFromSmiles(p))

    # get all possible combinations of reactants
    all_prds = set()
    for perms in permutations(reactants):
        try:
            products = rxn.RunReactants(perms)
            prd_smi = [process_product(i) for prd in products for i in prd]
            all_prds.update(prd_smi)
        except:
            pass

    return frozenset(all_prds)


# iteratively find all the 'mapped_reaction_smiles' items
def get_reactions(tree):
    if "is_reaction" in tree.keys():
        if tree["is_reaction"]:
            rxn = tree["metadata"]["mapped_reaction_smiles"]
            tmplt = tree["metadata"]["template"]
            rct = rxn.split(">>")[1]

            possible_prods = check_template(tmplt, rct)
            if "children" in tree.keys():
                result = {
                    "reaction": rxn,
                    "template": tmplt,
                    "result": "\n".join(possible_prods),
                    "len_prods": len(possible_prods),
                }

                return [result, [get_reactions(c) for c in tree["children"]]]
    else:
        if "children" in tree.keys():
            return [get_reactions(c) for c in tree["children"]]
        else:
            return 1


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        elif not isinstance(item, int):
            flat_list.append(item)
    return flat_list


def get_flag_tmplts(route):
    revise_rules = []
    for i in range(len(route)):
        res = flatten_list(get_reactions(route[i]))
        for r in res:
            if r["len_prods"] != 1:
                revise_rules.append(r)

    if len(revise_rules) == 0:
        return None
    df_tmplts = pd.DataFrame(revise_rules)
    return df_tmplts


if __name__ == "__main__":
    with open("mol1_output_zinc17m.json", "r") as f:
        route = json.load(f)["data"][0]["trees"]
        df_fix = get_flag_tmplts(route)
