import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

import litellm

from synthelite.agent_search.base import LLMBase
from synthelite.agent_search.schema import (
    PlanningAttempt,
    StrategicStep,
    SynthesisPlanUpdate,
    SynthesisContext,
)
from .strategy_prompt import (
    strategy_prompt_update_and_next_step_with_bond_id,
    search_results_evaluate_template,
    search_results_selection_template,
    search_results_selection_template_fallback,
    strategy_prompt_update_and_next_step_with_search,
)

from the_retry import retry
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SynthesisStrategyPlanner(LLMBase):
    """LLM-based synthesis strategy planner."""

    def create_update_strategy_and_next_step_prompt_with_search(
        self,
        target_smiles: str,
        user_constraint: str,
        previous_reactions: List[str],
        expandable_molecules: List[str],
        expandable_molecules_mapped: Optional[List[str]] = None,
        previous_attempts: List[Dict] = [],
        mol_solve_states: Optional[List[bool]] = None,
    ):
        def _join_smiles(
            molecules: List[str], states: Optional[List[bool]] = None
        ) -> List[str]:
            if states is None:
                return "\n".join(
                    [f"{i}. {smiles}" for i, smiles in enumerate(molecules)]
                )
            else:
                assert len(molecules) == len(
                    states
                ), "Length of molecules and states must match."
                return "\n".join(
                    [
                        f"{i}. {smiles} - {'[In-stock]' if state else '[Not in-stock]'}"
                        for i, (smiles, state) in enumerate(zip(molecules, states))
                    ]
                )

        previous_reaction_strs = [
            f"{i}. {reaction}" for i, reaction in enumerate(previous_reactions)
        ]

        if expandable_molecules_mapped is not None:
            expandable_molecules_str = _join_smiles(
                expandable_molecules, mol_solve_states
            )
            expandable_molecules_mapped_str = _join_smiles(
                expandable_molecules_mapped, mol_solve_states
            )

            return (
                strategy_prompt_update_and_next_step_with_bond_id.replace(
                    "{{TARGET_MOLECULE}}", target_smiles
                )
                .replace("{{USER_PROMPT}}", user_constraint)
                .replace(
                    "{{PREVIOUS_ATTEMPTS}}",
                    json.dumps(
                        [
                            PlanningAttempt.from_dict(each).to_string()
                            for each in previous_attempts
                        ],
                        indent=2,
                    ),
                )
                .replace("{{PREVIOUS_REACTIONS}}", "\n".join(previous_reaction_strs))
                .replace("{{CURRENT_MOLECULE_SMILES}}", expandable_molecules_str)
                .replace(
                    "{{CURRENT_MOLECULE_SMILES_MAPPED}}",
                    expandable_molecules_mapped_str,
                )
            )
        else:
            expandable_molecules_str = _join_smiles(
                expandable_molecules, mol_solve_states
            )
            return (
                strategy_prompt_update_and_next_step_with_search.replace(
                    "{{TARGET_MOLECULE}}", target_smiles
                )
                .replace("{{USER_PROMPT}}", user_constraint)
                .replace(
                    "{{PREVIOUS_ATTEMPTS}}",
                    json.dumps(
                        [
                            PlanningAttempt.from_dict(each).to_string()
                            for each in previous_attempts
                        ],
                        indent=2,
                    ),
                )
                .replace("{{PREVIOUS_REACTIONS}}", "\n".join(previous_reaction_strs))
                .replace("{{CURRENT_MOLECULE_SMILES}}", expandable_molecules_str)
            )

    def update_search_result_to_response(
        self,
        strategy_response: str,
        search_result: str,
        max_select: int = 1,
        mode: str = "verify",
        fallback_mode: bool = False,
    ):
        assert (
            strategy_response.count("<calling_search>") == 1
        ), "There should be exactly one <calling_search> tag in the strategy response."

        strategy_response = (
            strategy_response.partition("<calling_search>")[0] + "<calling_search>\n"
        )

        if mode == "verify":
            search_engine_response = search_results_evaluate_template.replace(
                "{{SEARCH_RESULT}}", search_result
            )
        elif mode == "select":
            template = (
                search_results_selection_template
                if not fallback_mode
                else search_results_selection_template_fallback
            )

            search_engine_response = template.replace(
                "{{SEARCH_RESULT}}", search_result
            )
            if max_select == 1:
                search_engine_response = search_engine_response.replace(
                    "{{MAX_SELECTS_REACTIONS}}", "the most appropriate reaction"
                )
            else:
                search_engine_response = search_engine_response.replace(
                    "{{MAX_SELECTS_REACTIONS}}",
                    f"up to {max_select} most appropriate reactions",
                )

        return strategy_response, search_engine_response

    def parse_strategy_response(self, response: str) -> dict:
        """Parse the LLM response and extract strategy JSON."""
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            return {
                "strategy_overview": "Could not parse structured strategy",
                "step_estimate": "Unknown",
                "steps": [],
                "additional_notes": response[:500],
            }
        except Exception as e:
            if self.verbose:
                print(f"Parse error: {e}")
            return {
                "strategy_overview": "Parse error occurred",
                "step_estimate": "Unknown",
                "steps": [],
                "error": str(e),
                "raw_content": response[:500],
            }

    def _update_regex_key_to_dict(self, key: str, raw_text: str, dict_obj: dict):
        match = re.search(rf"<{key}>(.*?)</{key}>", raw_text, re.DOTALL)
        try:
            dict_obj[key] = match.group(1).strip()
        except Exception as e:
            if self.verbose:
                print(f"Next step parse error: {e}")
            dict_obj[key] = None

    def parse_strategy_with_next_step_response(self, response: str):
        strategy = self.parse_strategy_response(response)

        self._update_regex_key_to_dict("next_retro_transformation", response, strategy)
        self._update_regex_key_to_dict("next_forward_reaction", response, strategy)
        self._update_regex_key_to_dict("expandable_molecule_index", response, strategy)
        self._update_regex_key_to_dict("reaction_atom_indices", response, strategy)
        self._update_regex_key_to_dict("is_ring_break", response, strategy)

        strategy["expandable_molecule_index"] = (
            int(strategy["expandable_molecule_index"])
            if strategy["expandable_molecule_index"] is not None
            else None
        )

        strategy["reaction_atom_indices"] = (
            json.loads(strategy["reaction_atom_indices"])
            if strategy["reaction_atom_indices"] is not None
            else None
        )

        strategy["is_ring_break"] = (
            True if strategy["is_ring_break"].lower() == "true" else False
        )

        return strategy

    def _run_sync(self, coro):
        """Execute an async coroutine synchronously, handling running loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            return loop.run_until_complete(coro)

    def create_strategy(self, target_smiles: str, user_constraint: str) -> Dict:
        """Generate a synthesis strategy using the LLM."""
        if self.verbose:
            print(f"Creating synthesis strategy for target: {target_smiles[:50]}")
            print(f"User constraint: {user_constraint}")

        prompt_text = self.create_strategy_prompt(target_smiles, user_constraint)
        if self.verbose:
            print(f"Prompt created ({len(prompt_text)} characters)")

        raw_response = self._run_sync(
            self._query_llm(messages=[{"role": "user", "content": prompt_text}])
        )
        if self.verbose:
            print(f"LLM response received ({len(raw_response)} characters)")

        parsed_strategy = self.parse_strategy_response(raw_response)
        if self.verbose:
            print("Strategy parsed successfully.")

        parsed_strategy.update(
            {
                "target_smiles": target_smiles,
                "user_constraint": user_constraint,
                "model_used": self.model,
                "raw_response": raw_response,
            }
        )

        return parsed_strategy

    def update_strategy(
        self,
        target_smiles: str,
        user_constraint: str,
        previous_reactions: List[str],
        expandable_molecules: List[str],
    ):
        """Update an existing synthesis strategy with new information."""
        if self.verbose:
            print(f"Updating synthesis strategy for target: {target_smiles[:50]}")
            print(f"User constraint: {user_constraint}")
            print(f"Previous reactions: {previous_reactions}")
            print(f"Expanded molecules: {expandable_molecules}")

        prompt_text = self.create_update_strategy_prompt(
            target_smiles, user_constraint, previous_reactions, expandable_molecules
        )

        if self.verbose:
            print(f"Update prompt created ({len(prompt_text)} characters)")

        raw_response = self._run_sync(
            self._query_llm(messages=[{"role": "user", "content": prompt_text}])
        )
        if self.verbose:
            print(f"LLM response received ({len(raw_response)} characters)")

        parsed_strategy = self.parse_strategy_response(raw_response)
        if self.verbose:
            print("Strategy updated successfully.")

        response = {
            "target_smiles": target_smiles,
            "user_constraint": user_constraint,
            "previous_reactions": previous_reactions,
            "expandable_molecules": expandable_molecules,
            "synthesis_plan": parsed_strategy,
            "model_used": self.model,
            "raw_response": raw_response,
        }

        return SynthesisPlanUpdate.from_dict(response)

    def has_stop_signal(self, raw_response: str):
        return bool(
            re.search(
                r"<stop_signal>\s*TRUE\s*</stop_signal>", raw_response, re.IGNORECASE
            )
        )

    @retry(attempts=3, backoff=5)
    async def update_strategy_and_propose_next_step_with_search(
        self,
        target_smiles: str,
        user_constraint: str,
        previous_reactions: List[str],
        expandable_molecules: List[str],
        expandable_molecules_mapped: Optional[List[str]] = None,
        previous_attempts: List[Dict] = [],
        mol_solve_states: Optional[List[bool]] = None,
    ):
        if expandable_molecules_mapped is not None:
            assert len(expandable_molecules) == len(
                expandable_molecules_mapped
            ), "Length of expandable_molecules and expandable_molecules_mapped must match."

        if self.verbose:
            print(f"Updating synthesis strategy for target: {target_smiles[:50]}")
            print(f"User constraint: {user_constraint}")
            print(f"Previous reactions: {previous_reactions}")
            print(f"Expanded molecules: {expandable_molecules}")

        prompt_text = self.create_update_strategy_and_next_step_prompt_with_search(
            target_smiles,
            user_constraint,
            previous_reactions,
            expandable_molecules,
            expandable_molecules_mapped,
            previous_attempts=previous_attempts,
            mol_solve_states=mol_solve_states,
        )

        if self.verbose:
            print(f"Update prompt created ({len(prompt_text)} characters)")

        raw_response = await self._query_llm(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        )
        if self.verbose:
            print(f"LLM response received ({len(raw_response)} characters)")

        if self.has_stop_signal(raw_response):
            return None

        response = self.parse_strategy_with_next_step_response(raw_response)
        if self.verbose:
            print("Strategy updated successfully.")

        response.update(
            {
                "target_smiles": target_smiles,
                "user_constraint": user_constraint,
                "previous_reactions": previous_reactions,
                "expandable_molecules": expandable_molecules,
                "expandable_molecules_mapped": expandable_molecules_mapped,
                "mol_solve_states": mol_solve_states,
                "synthesis_context": {
                    "target_smiles": target_smiles,
                    "user_constraint": user_constraint,
                    "previous_attempts": previous_attempts,
                },
                "model_used": self.model,
                "raw_response": raw_response,
            }
        )
        assert (
            "next_forward_reaction" in response
        ), "Next forward reaction not found in response."

        res = SynthesisPlanUpdate.from_dict(response)

        res = self._correct_synthesis_plan(res, previous_reactions)
        return res

    @retry(attempts=3, backoff=5)
    async def validate_found_reaction(
        self, strategy_response: SynthesisPlanUpdate, reaction: str
    ):
        model_response, tool_response = self.update_search_result_to_response(
            strategy_response.raw_response, reaction, mode="verify"
        )

        previous_attempts = [
            each.to_dict() if not isinstance(each, dict) else each
            for each in strategy_response.synthesis_context.previous_attempts
        ]

        user_prompt = self.create_update_strategy_and_next_step_prompt_with_search(
            target_smiles=strategy_response.target_smiles,
            user_constraint=strategy_response.user_constraint,
            previous_reactions=strategy_response.previous_reactions,
            previous_attempts=previous_attempts,
            expandable_molecules=strategy_response.expandable_molecules,
            expandable_molecules_mapped=strategy_response.expandable_molecules_mapped,
            mol_solve_states=strategy_response.mol_solve_states,
        )

        raw_response = await self._query_llm(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": model_response,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": tool_response},
            ]
        )

        verdict_match = re.search(r"<verdict>(.*?)</verdict>", raw_response, re.DOTALL)

        try:
            verdict = verdict_match.group(1).strip().strip('"').lower()
            if verdict not in ["accept", "reject"]:
                raise ValueError(
                    f"Invalid verdict: {verdict}. Expected 'accept' or 'reject'."
                )
        except Exception as e:
            verdict = "accept" if "accept" in raw_response.lower() else "reject"

        return verdict == "accept"

    @retry(attempts=3, backoff=5)
    async def select_found_reactions(
        self,
        strategy_response: SynthesisPlanUpdate,
        reactions: List[str],
        max_select: int = 1,
        fallback_mode: bool = False,
    ):
        search_results = "\n".join([f"{i}. {rxn}" for i, rxn in enumerate(reactions)])
        model_response, tool_response = self.update_search_result_to_response(
            strategy_response.raw_response,
            search_results,
            max_select=max_select,
            mode="select",
            fallback_mode=fallback_mode,
        )

        previous_attempts = [
            each.to_dict() if not isinstance(each, dict) else each
            for each in strategy_response.synthesis_context.previous_attempts
        ]

        user_prompt = self.create_update_strategy_and_next_step_prompt_with_search(
            target_smiles=strategy_response.target_smiles,
            user_constraint=strategy_response.user_constraint,
            previous_reactions=strategy_response.previous_reactions,
            previous_attempts=previous_attempts,
            expandable_molecules=strategy_response.expandable_molecules,
            expandable_molecules_mapped=strategy_response.expandable_molecules_mapped,
            mol_solve_states=strategy_response.mol_solve_states,
        )

        raw_response = await self._query_llm(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": model_response,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": tool_response},
            ]
        )

        selected_indices_match = re.search(
            r"<selected_reaction_indices>(.*?)</selected_reaction_indices>",
            raw_response,
            re.DOTALL,
        )

        if not selected_indices_match:
            raise ValueError(
                "Could not find selected reaction indices in the response."
            )

        selected_indices = json.loads(selected_indices_match.group(1).strip())

        return selected_indices[:max_select]

    @retry(attempts=3, backoff=5)
    async def update_strategy_and_propose_next_step(
        self,
        target_smiles: str,
        user_constraint: str,
        previous_reactions: List[str],
        expandable_molecules: List[str],
    ):
        if self.verbose:
            print(f"Updating synthesis strategy for target: {target_smiles[:50]}")
            print(f"User constraint: {user_constraint}")
            print(f"Previous reactions: {previous_reactions}")
            print(f"Expanded molecules: {expandable_molecules}")

        prompt_text = self.create_update_strategy_and_next_step_prompt(
            target_smiles, user_constraint, previous_reactions, expandable_molecules
        )

        if self.verbose:
            print(f"Update prompt created ({len(prompt_text)} characters)")

        raw_response = await self._query_llm(
            messages=[{"role": "user", "content": prompt_text}]
        )
        if self.verbose:
            print(f"LLM response received ({len(raw_response)} characters)")

        response = self.parse_strategy_with_next_step_response(raw_response)
        if self.verbose:
            print("Strategy updated successfully.")

        response.update(
            {
                "target_smiles": target_smiles,
                "user_constraint": user_constraint,
                "previous_reactions": previous_reactions,
                "expandable_molecules": expandable_molecules,
                "model_used": self.model,
                "raw_response": raw_response,
            }
        )

        res = SynthesisPlanUpdate.from_dict(response)

        res = self._correct_synthesis_plan(res, previous_reactions)
        return res

    def _correct_synthesis_plan(
        self, synthesis_plan: SynthesisPlanUpdate, previous_reactions
    ):
        """Ensure that the synthesis plan has the correct number of previous steps."""
        if synthesis_plan.previous_steps is not None and len(previous_reactions) == len(
            synthesis_plan.previous_steps
        ):
            for reaction, previous_step in zip(
                previous_reactions, synthesis_plan.previous_steps
            ):
                previous_step.step_reaction = reaction
        else:
            logger.warning(
                "Previous reactions and previous steps length mismatch. Rebuilding previous steps."
            )
            previous_steps = []
            for i, reaction in enumerate(previous_reactions):
                step = StrategicStep(step_number=i + 1, step_reaction=reaction)
                previous_steps.append(step)

            synthesis_plan.previous_steps = previous_steps

        return synthesis_plan
