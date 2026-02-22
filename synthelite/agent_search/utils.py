from typing import Optional
from pandas import DataFrame

from synthelite.agent_search.schema import PlanningAttempt, StrategicRoute
from synthelite.chem.reaction import AbsoluteRetroReaction
from synthelite.search.mcts.node import LLMGuidedMctsNode
from synthelite.search.mcts.search import LLMGuidedBeamSearchTree


def get_step_description(
    action: AbsoluteRetroReaction,
    template_df: Optional[DataFrame] = None,
    use_llm_query: bool = True,
) -> str:
    if "lmdata" in action.metadata and use_llm_query:
        return action.metadata.get("lmdata", {}).get("query", "")
    elif "description" in action.metadata:
        return action.metadata.get("description", "")
    elif "template_code" in action.metadata and template_df is not None:
        template_code = int(action.metadata["template_code"])
        template_row = template_df[template_df["template_code"] == template_code]
        if not template_row.empty:
            return template_row.llm_description.item()

    else:
        return action.metadata.get("query", "")


def summarize_leaf(leaf: LLMGuidedMctsNode, attempt_id: Optional[str] = None) -> dict:
    actions, nodes = leaf.path_to()

    steps = []
    for i, action in enumerate(actions):
        product = action.mol.smiles
        product_mapped = action.mol.mapped_smiles
        reactants = [mol.smiles for mol in action.reactants[0]]
        reactants_mapped = [mol.mapped_smiles for mol in action.reactants[0]]
        reactants = ".".join(reactants)
        reactants_mapped = ".".join(reactants_mapped)

        retro_reaction_smi = f"{product}>>{reactants}"
        retro_reaction_mapped_smi = f"{product_mapped}>>{reactants_mapped}"

        steps.append(retro_reaction_smi)

    unsolved_molecules = [mol.smiles for mol in leaf.state.expandable_mols]
    is_solved = leaf.is_solved

    res_dict = {
        "attempt_id": attempt_id,
        "retro_reactions": steps,
        "is_solved": is_solved,
        "unsolved_molecules": unsolved_molecules,
    }

    return PlanningAttempt.from_dict(res_dict)


def get_representative_node(tree: LLMGuidedBeamSearchTree):
    """Get the first solved node or the first leaf node (dfs) if no solved node is found.


    Args:
        tree (LLMGuidedBeamSearchTree): _description_

    Returns:
        _type_: _description_
    """
    for leaf in tree.get_all_leaf_nodes(include_expanded=True):
        if leaf.is_solved:
            return leaf
    return tree.get_all_leaf_nodes(include_expanded=True)[0]


def summarize_tree(
    tree: LLMGuidedBeamSearchTree, attempt_id: Optional[str] = None
) -> StrategicRoute:
    representative_node = get_representative_node(tree)
    return summarize_leaf(representative_node, attempt_id)
