"""
LLM 3: Strategy Alignment Scorer
Individual scoring of children against synthesis strategy
"""
from synthelite.agent_search.strategy_alignment_scorer_prompt import (
    scoring_prompt,
    scoring_prompt_old,
    ranking_prompt,
)
import litellm
import json
from typing import List, Dict, Any, Tuple, Union
import re
import logging

from synthelite.agent_search.schema import (
    StrategyAlignmentScorerResponse,
    SynthesisContext,
    StrategicRoute,
)

from the_retry import retry

logger = logging.getLogger(__name__)


def _extract_score_and_justification(raw_response: str):
    """Extract score and justification from an LLM response."""
    score_match = re.search(
        r"<score>\s*([0-9\.]+)\s*</score>", raw_response, re.DOTALL | re.IGNORECASE
    )
    justification_match = re.search(
        r"<justification>(.*?)</justification>", raw_response, re.DOTALL | re.IGNORECASE
    )
    score = float(score_match.group(1)) if score_match else None
    justification = (
        justification_match.group(1).strip()
        if justification_match
        else "No justification provided"
    )
    return score, justification


class StrategicRouteScorer:
    def __init__(
        self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1
    ):
        """Initialize the scorer."""
        self.model = model
        self.temperature = temperature

    def extract_route_evaluation_and_rank(
        self, raw_response: str, input_routes: List[StrategicRoute]
    ) -> List[StrategicRoute]:
        """Parse optional route evaluation and robustly assign a rank to each route.

        - If <route_evaluation> JSON is present and parseable, merge any scores
          back into the route objects.
        - If <route_ranking> JSON is present and valid, use it as the canonical
          order (IDs are indices in the input list).
        - Otherwise, fall back to ordering by `route_score` when available;
          if no scores are available, preserve original order.
        """
        res_routes = input_routes.copy()

        # 1) Optional: merge route evaluation (scores) if present
        route_eval_match = re.search(
            r"<route_evaluation>(.*?)</route_evaluation>",
            raw_response,
            re.DOTALL | re.IGNORECASE,
        )
        if route_eval_match:
            route_eval_txt = route_eval_match.group(1).strip()
            try:
                evaluated_routes = [
                    StrategicRoute.from_dict(route)
                    for route in json.loads(route_eval_txt)
                ]
                evaluated_routes = [
                    route for route in evaluated_routes if route.route_score is not None
                ]
                for route in evaluated_routes:
                    route_id = int(route.route_id)
                    res_routes[route_id] = route
            except Exception:
                logger.warning("Failed to parse route evaluation. Use original routes")

        # 2) Rank routes: prefer explicit <route_ranking>, else use score-based fallback
        ranking_match = re.search(
            r"<route_ranking>(.*?)</route_ranking>",
            raw_response,
            re.DOTALL | re.IGNORECASE,
        )

        order_ids: List[int] | None = None
        if ranking_match:
            try:
                order_ids = json.loads(ranking_match.group(1).strip())
                order_ids = [int(each) for each in order_ids]
            except Exception:
                logger.warning(
                    "Failed to parse route ranking. Falling back to score-based or original order"
                )
                order_ids = None

        if order_ids is None:
            # Fallback: by route_score if any present, else original index order
            has_any_score = any(
                getattr(r, "route_score", None) is not None for r in res_routes
            )
            if has_any_score:
                order_indices = sorted(
                    range(len(res_routes)),
                    key=lambda i: (res_routes[i].route_score or 0),
                    reverse=True,
                )
                for rank, idx in enumerate(order_indices):
                    res_routes[idx].route_rank = rank + 1
            else:
                for rank, _ in enumerate(res_routes):
                    res_routes[rank].route_rank = rank + 1
        else:
            id_to_rank = {i: rank for rank, i in enumerate(order_ids)}
            for i, route in enumerate(res_routes):
                try:
                    route_id = int(route.route_id)
                except Exception:
                    route_id = i
                if route_id in id_to_rank:
                    route.route_rank = id_to_rank[route_id] + 1
                else:
                    # If not present in returned ranking, deprioritize but keep finite
                    route.route_rank = len(res_routes) + 1

        return res_routes

    @retry(attempts=3, backoff=5)
    def rank_routes(
        self,
        synthesis_context: SynthesisContext,
        routes: List[StrategicRoute],
    ):
        prompt = ranking_prompt.replace(
            "{{SYNTHESIS_STRATEGY}}", synthesis_context.to_string()
        )
        prompt = prompt.replace(
            "{{CANDIDATE_ROUTES}}",
            json.dumps([route.to_dict() for route in routes], indent=2),
        )

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        raw_response = response.choices[0].message.content.strip()
        evaluated_routes = self.extract_route_evaluation_and_rank(raw_response, routes)
        return evaluated_routes


class StrategyAlignmentScorer:
    """LLM 3: Strategy Alignment Scorer
    Individual scoring of children against synthesis strategy
    """

    def __init__(
        self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.1
    ):
        """Initialize the scorer."""
        self.model = model
        self.temperature = temperature

    def score_single_child(
        self,
        smiles_path: List[str],
        strategy: Dict[str, Any],
        current_depth: int = None,
    ) -> Tuple[float, str]:
        """Score a single child against the synthesis strategy."""

        # Create prompt
        prompt = scoring_prompt.replace("{{PREVIOUS_SMILES}}", "\n".join(smiles_path))
        prompt = prompt.replace(
            "{{SYNTHESIS_STRATEGY}}", json.dumps(strategy, indent=2)
        )

        # Call LLM
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        raw_response = response.choices[0].message.content.strip()

        try:
            score, justification = _extract_score_and_justification(raw_response)
            if score is not None:
                return StrategyAlignmentScorerResponse(
                    score=score,
                    justification=justification,
                    raw_response=raw_response,
                    prompt=prompt,
                )
        except ValueError:
            pass

        return StrategyAlignmentScorerResponse(
            score=0.0,
            justification="Failed to parse score and justification",
            raw_response=raw_response,
            prompt=prompt,
        )

    def score_action_with_context(
        self,
        action,
        current_path: List[str],
        strategy: Union[Dict, SynthesisContext],
        current_depth: int,
        reaction_mapped: bool = False,
        retro: bool = True,
    ) -> Tuple[float, str]:
        """Score an action based on the reaction path and synthesis strategy.

        current_depth is ignored for now, since it can be inferred from len(current_path).
        """

        if not action.reactants:
            return 0.0, "No reactants generated"

        strategy = (
            strategy
            if isinstance(strategy, SynthesisContext)
            else SynthesisContext.from_dict(strategy)
        )

        if reaction_mapped:
            product_smiles = action.mol.mapped_smiles
            reactants_smiles = ".".join([r.mapped_smiles for r in action.reactants[0]])
        else:
            product_smiles = action.mol.smiles
            reactants_smiles = ".".join([r.smiles for r in action.reactants[0]])

        if retro:
            # reverse the reaction for retrosynthesis
            new_reaction = f"{product_smiles}>>{reactants_smiles}"
        else:
            new_reaction = f"{reactants_smiles}>>{product_smiles}"

        # extend path
        extended_path = current_path + [new_reaction]

        prompt = scoring_prompt.replace("{{PREVIOUS_SMILES}}", "\n".join(extended_path))
        prompt = prompt.replace(
            "{{SYNTHESIS_STRATEGY}}", json.dumps(strategy.__dict__, indent=2)
        )
        prompt = prompt.replace("{{CURRENT_DEPTH}}", str(current_depth + 1))

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            num_retries=3,
        )

        raw_response = response.choices[0].message.content.strip()

        try:
            score, justification = _extract_score_and_justification(raw_response)
            if score is not None:
                return StrategyAlignmentScorerResponse(
                    score=score,
                    justification=justification,
                    raw_response=raw_response,
                    prompt=prompt,
                )
            else:
                logger.warning(
                    "No score found in LLM response: %s...", raw_response[:200]
                )
                return StrategyAlignmentScorerResponse(
                    score=5.0,
                    justification="Could not parse score from LLM response",
                    raw_response=raw_response,
                    prompt=prompt,
                )

        except Exception as e:
            logger.error("Error in score_action_with_context: %s", e)
            return StrategyAlignmentScorerResponse(
                score=0.5,
                justification=f"Error: {e}",
                raw_response=raw_response,
                prompt=prompt,
            )
