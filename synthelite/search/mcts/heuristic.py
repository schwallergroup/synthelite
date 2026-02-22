import random
from typing import List, TYPE_CHECKING, Optional
import numpy as np
import functools

from synthelite.agent_search.schema import (
    StrategicRoute,
    StrategicStep,
    StrategyAlignmentScorerResponse,
    SynthesisContext,
)
from synthelite.agent_search.strategy_alignment_scorer import (
    StrategyAlignmentScorer,
    StrategicRouteScorer,
)
from synthelite.chem.reaction import action_to_str
from synthelite.utils.clogging import init_weave
import weave

if TYPE_CHECKING:
    from synthelite.search.mcts.node import LLMGuidedMctsNode
    from synthelite.context.config import Configuration
init_weave()


def fixed_strategy_heuristic_score(node: "LLMGuidedMctsNode"):
    """Return a list of LLM-scored heuristic scores for the children of this node.
    This function is called after node._children_actions has been initialized.
    """
    if node._children_heuristic_scores:
        return np.array(node._children_heuristic_scores, dtype=float)
    scores = [0.0] * len(node._children_actions)
    responses = [None] * len(node._children_actions)

    llm_config = node._config.search.llm_guidance.strategic_scorer
    scorer = StrategyAlignmentScorer(
        model=llm_config.get("model", "claude-3-5-sonnet-20241022"), temperature=0.1
    )

    current_path = node.get_smiles_tree()
    strategy = getattr(node.tree, "_synthesis_plan", None)

    for i, action in enumerate(node._children_actions):
        if not action or not action.reactants:
            continue
        scorer_response: StrategyAlignmentScorerResponse = (
            scorer.score_action_with_context(
                action=action,
                current_path=current_path,
                strategy=strategy,
                current_depth=len(current_path),
            )
        )
        alignment_score = scorer_response.score
        justification = scorer_response.justification
        scores[i] = alignment_score
        # responses[i] = {'response': justification, 'score': float(alignment_score)}
        responses[i] = scorer_response.__dict__

    for i, action in enumerate(node._children_actions):
        action.metadata["lmdata"] = {
            "query": "LLM_Strategy_Alignment",
            "response": responses[i]["justification"] if responses[i] else None,
            "lm_score": scores[i],
            "prompt": responses[i].get("prompt") if responses[i] else None,
            "raw_response": responses[i].get("raw_response") if responses[i] else None,
        }

    node._children_heuristic_scores = scores
    result = np.array(scores, dtype=float)
    return result


@weave.op()
def dynamic_strategy_heuristic_rank(
    nodes: List["LLMGuidedMctsNode"], repetitions: int = 1
):
    """
    Use a dynamic strategy that is updated at every step.
    Also ranking the nodes instead of scoring.
    """
    # if node._children_heuristic_scores:
    #     return np.array(node._children_heuristic_scores, dtype=float)
    n_children = sum(len(node.children) for node in nodes)
    if n_children == 0:
        return [[]] * len(nodes)

    if n_children == 1:  # Only rank if there are multiple children
        return [[1.0]]

    scores = [0.0] * n_children
    # responses = [None] * len(node._children_actions)

    llm_config = nodes[0]._config.search.llm_guidance.strategic_scorer
    scorer = StrategicRouteScorer(
        model=llm_config.get("model", "claude-3-5-sonnet-20241022"), temperature=0.1
    )

    synthesis_context = SynthesisContext(
        target_smiles=nodes[0].target_molecule.smiles,
        user_constraint=nodes[0].user_constraint,
    )

    rank_reps = []
    for rep in range(repetitions):
        candidate_children_routes, true_route_ids = (
            _get_children_strategic_routes_multi_nodes(nodes)
            if rep == 0
            else _get_children_strategic_routes_multi_nodes(nodes, permute=True)
        )

        scored_routes = scorer.rank_routes(
            synthesis_context=synthesis_context, routes=candidate_children_routes
        )

        # route_ids = [each.route_id for each in scored_routes]
        ranks = [each.route_rank for each in scored_routes]
        # replace None value with 0
        ranks = [rank if rank is not None else 0 for rank in ranks]
        true_route_id_to_rank = {
            route_id: rank for route_id, rank in zip(true_route_ids, ranks)
        }
        ranks = [
            true_route_id_to_rank[true_id] for true_id in range(len(true_route_ids))
        ]
        rank_reps.append(ranks)

    ranks_average = np.mean(rank_reps, axis=0)

    for i, rank in enumerate(ranks_average):
        scores[i] = 1 / rank if rank > 0 else 0.0  # Avoid division by zero

    # for route_id, rank in zip(route_ids, ranks):
    #     scores[int(route_id)] = 1 / rank

    # for i, action in enumerate(node._children_actions):
    #     action.metadata["lmdata"] = {
    #         "query": "Dynamic_Strategy_Route_Ranking",
    #         # "response": responses[i]["justification"] if responses[i] else None,
    #         "lm_score": scores[i],
    #         # "prompt": responses[i].get("prompt") if responses[i] else None,
    #         # "raw_response": responses[i].get("raw_response") if responses[i] else None,
    #     }

    # node._children_heuristic_scores = scores
    current_action_id = 0
    for node in nodes:
        node_children_heuristic_scores = [0.0] * len(node._children)
        children_indices = [i for i, child in enumerate(node._children) if child]
        node_scores = scores[
            current_action_id : current_action_id + len(children_indices)
        ]
        for i, child_idx in enumerate(children_indices):
            node_children_heuristic_scores[child_idx] = node_scores[i]
        node._children_heuristic_scores = node_children_heuristic_scores
        # for i, action in enumerate(node._children_actions):
        #     action.metadata["lmdata"] = {
        #         "query": "Dynamic_Strategy_Route_Ranking",
        #         "lm_score": scores[current_action_id + i],
        #     }
        current_action_id += len(children_indices)

    # result = np.array(scores, dtype=float)
    return [
        [
            score
            for child, score in zip(each._children, each._children_heuristic_scores)
            if child is not None
        ]
        for each in nodes
    ]


def _get_children_strategic_routes_per_node(
    node: "LLMGuidedMctsNode", start_route_id: int = 0
) -> List[StrategicRoute]:
    if not node.synthesis_dynamic_context or not node._children_actions:
        return []
    previous_steps: List[StrategicStep] = node.synthesis_dynamic_context.previous_steps
    next_step_number = len(previous_steps) + 1
    next_steps = node.synthesis_dynamic_context.next_steps
    if next_steps and isinstance(next_steps[0], dict):
        next_steps = [StrategicStep.from_dict(step) for step in next_steps]
    # next_step_description = next_step.get('step_description', None) if isinstance(next_step, dict) else next_step.step_description
    routes: List[StrategicRoute] = []
    # true_route_indices = []

    for child_idx, child in enumerate(node.children):
        reaction_pathway = child.get_smiles_tree()
        if not reaction_pathway:
            continue
        reaction_str = reaction_pathway[-1]

        # Create a new strategic step for this action
        next_steps_emphemeral = []

        for step_id, step in enumerate(next_steps):
            if step_id == 0:
                next_steps_emphemeral.append(
                    StrategicStep(
                        step_number=next_step_number + step_id,
                        step_reaction=reaction_str,
                        step_description=step.step_description,
                    )
                )
            else:
                next_steps_emphemeral.append(
                    StrategicStep(
                        step_number=next_step_number + step_id,
                        step_reaction=None,
                        step_description=step.step_description,
                    )
                )

        steps = previous_steps + next_steps_emphemeral
        # Create a new strategic route
        route = StrategicRoute(
            route_id=str(start_route_id + child_idx),
            steps=steps,
        )
        # true_child_index = node._children.index(child)

        routes.append(route)
        # true_route_indices.append(true_child_index)

    return routes


def _get_children_strategic_routes_multi_nodes(
    nodes: List["LLMGuidedMctsNode"], permute=False
) -> List[StrategicRoute]:
    """
    Get strategic routes for multiple nodes.
    """
    routes = []
    # true_node_child_indices = []
    # node_and_child_id_to_route = {}
    start_route_id = 0
    for node in nodes:
        node_routes = _get_children_strategic_routes_per_node(node, start_route_id)
        # true_node_child_indices.extend([(node, idx) for idx in true_child_indices])
        # for node_route, true_child_id in zip(node_routes, true_child_indices):
        #     node_and_child_id_to_route[(node, true_child_id)] = node_route
        routes.extend(node_routes)
        start_route_id += len(node_routes)

    true_route_indices = [int(route.route_id) for route in routes]

    if permute:
        random.shuffle(routes)
        true_route_indices = [int(route.route_id) for route in routes]

        for new_route_id, route in enumerate(routes):
            route.route_id = str(new_route_id)

    return routes, true_route_indices


HEURISTIC_SCORER = {
    "fixed_strategy_heuristic_score": fixed_strategy_heuristic_score,
    "dynamic_strategy_heuristic_rank": dynamic_strategy_heuristic_rank,
}


def get_route_heuristic_score_fn(config: "Configuration"):
    fn_name = config.search.llm_guidance.strategic_scorer["score_function"]
    fn_args = config.search.llm_guidance.strategic_scorer.get("score_args", {})

    fn = HEURISTIC_SCORER[fn_name]

    return functools.partial(fn, **fn_args) if fn_args else fn
