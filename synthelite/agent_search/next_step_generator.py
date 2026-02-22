"""
Next-Step transformation generator: generates disconnection suggestions
"""
import litellm
import re
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from synthelite.agent_search.next_step_generator_prompt import (
    next_step_prompt,
    next_step_prompt_for_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class NextStepReponse:
    guide: str
    raw_response: str
    prompt: str


class NextStepGeneratorBase(ABC):
    def __init__(
        self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.3
    ):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def predict_next_step(self):
        raise NotImplementedError()


class NextStepGenerator(NextStepGeneratorBase):
    """LLM 2: Next-Step generator"""

    def predict_next_step(
        self,
        current_smiles: str,
        reaction_path: List[str],
        current_depth: int,
        strategy: Dict,
    ) -> str:
        """Generate disconnection suggestion for next step based on current depth, reaction path and synthesis strategy."""

        if not strategy:
            # Try to retrieve stored strategy
            raise ValueError(
                "No synthesis strategy provided or stored. Please provide a valid strategy."
            )

        # Format reaction path
        if not reaction_path:
            path_text = "Starting from target molecule (no previous steps)"
        else:
            path_text = "\n".join(
                [f"Step {i+1}: {rxn}" for i, rxn in enumerate(reaction_path)]
            )

        strategy_keys = [
            "user_constraint",
            "strategy_overview",
            "steps",
            "additional_notes",
        ]
        strategy = {k: strategy[k] for k in strategy_keys}
        # Create prompt
        filled_prompt = next_step_prompt.replace(
            "{{CURRENT_MOLECULE_SMILES}}", current_smiles
        )
        filled_prompt = filled_prompt.replace(
            "{{PARTIALLY_CONSTRUCTED_ROUTE_SMILES}}", path_text
        )
        filled_prompt = filled_prompt.replace("{{SYNTHESIS_DEPTH}}", str(current_depth))
        filled_prompt = filled_prompt.replace(
            "{{SYNTHESIS_STRATEGY}}", json.dumps(strategy, indent=2)
        )

        # Call LLM
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": filled_prompt}],
            temperature=self.temperature,
        )

        raw_response = response.choices[0].message.content

        # Extract disconnection
        try:
            match = re.search(
                r"<next_disconnection>(.*?)</next_disconnection>",
                raw_response,
                re.DOTALL,
            )
            if match:
                return match.group(1).strip()
            else:
                # Fallback to last line
                return raw_response.strip().split("\n")[-1].strip()
        except Exception:
            return "Disconnect the most reactive functional group through a standard transformation."


class NextStepGeneratorForBatch(NextStepGeneratorBase):
    def predict_next_step(
        self,
        target_molecule: str,
        current_smiles_batch: List[str],
        reaction_path: List[str],
        strategy: Dict,
    ) -> str:

        """
        Generate disconnection suggestion for next step based on current depth, reaction path and synthesis strategy.
        Instead of using a single molecule, this method is given the current state of the synthesis (a set of molecules) as input.
        Since we are working with the state of the synthesis, there is no need for current_depth.
        """

        if not strategy:
            # Try to retrieve stored strategy
            raise ValueError(
                "No synthesis strategy provided or stored. Please provide a valid strategy."
            )

        # Format reaction path
        if not reaction_path:
            path_text = "Starting from target molecule (no previous steps)"
        else:
            path_text = "\n".join(
                [f"Step {i+1}: {rxn}" for i, rxn in enumerate(reaction_path)]
            )

        filled_prompt = next_step_prompt_for_batch.replace(
            "{{TARGET_MOLECULE_SMILES}}", target_molecule
        )
        filled_prompt = filled_prompt.replace(
            "{{CURRENT_MOLECULE_SMILES}}", "\n".join(current_smiles_batch)
        )
        filled_prompt = filled_prompt.replace(
            "{{PARTIALLY_CONSTRUCTED_ROUTE_SMILES}}", path_text
        )
        filled_prompt = filled_prompt.replace(
            "{{SYNTHESIS_STRATEGY}}", json.dumps(strategy, indent=2)
        )

        # Call LLM
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": filled_prompt}],
            temperature=self.temperature,
            num_retries=3,
        )

        raw_response = response.choices[0].message.content

        # Extract disconnection
        try:
            match = re.search(
                r"<next_disconnection>(.*?)</next_disconnection>",
                raw_response,
                re.DOTALL,
            )
            if match:
                guide = match.group(1).strip()
            else:
                # Fallback to last line
                guide = raw_response.strip().split("\n")[-1].strip()
        except Exception:
            guide = "Disconnect the most reactive functional group through a standard transformation."

        res = NextStepReponse(
            guide=guide, raw_response=raw_response, prompt=filled_prompt
        )

        return res
