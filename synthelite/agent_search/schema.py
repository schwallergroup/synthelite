from dataclasses import dataclass
import json
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel


class DataModel(BaseModel):
    """Base model for data validation and serialization."""

    @classmethod
    def from_dict(cls, data_dict: Dict):
        """Create an instance from a dictionary."""
        return cls.model_validate(data_dict)

    def to_dict(self) -> Dict:
        """Convert the instance to a dictionary."""
        return self.model_dump(exclude_none=True)

    def to_string(self, indent: int = 2) -> str:
        """Convert the instance to a formatted string."""
        return self.model_dump_json(indent=indent, exclude_none=True)


class StrategicStep(DataModel):
    step_number: int
    step_reaction: Optional[str] = None
    step_description: Optional[str] = None
    step_evaluation: Optional[str] = None
    step_score: Optional[float] = None


class StrategicRoute(DataModel):
    route_id: Union[str, int]
    steps: List[StrategicStep]
    route_evaluation: Optional[str] = None
    route_score: Optional[float] = None
    route_rank: Optional[int] = None
    is_solved: Optional[bool] = None
    unsolved_molecules: Optional[List[str]] = None
    terminate_reason: Optional[str] = None
    feedback: Optional[str] = None


class StepFeedback(DataModel):
    step_id: int
    feedback: Optional[str] = None


class RouteFeedback(DataModel):
    overall_feedback: Optional[str] = None
    problematic_steps: Optional[List[StepFeedback]] = None


class PlanningAttempt(DataModel):
    attempt_id: Union[str, int]
    retro_reactions: List[str]
    is_solved: bool = False
    unsolved_molecules: List[str] = []
    feedback: Optional[Union[str, RouteFeedback]] = None

    def to_string(self) -> str:
        reaction_strs = [
            f"{i}. {reaction}" for i, reaction in enumerate(self.retro_reactions)
        ]
        reaction_strs = "\n".join(reaction_strs)
        unsolved_molecules_str = "\n".join(self.unsolved_molecules)
        feedback_str = (
            json.dumps(self.feedback.to_dict(), indent=2)
            if isinstance(self.feedback, RouteFeedback)
            else self.feedback
        )
        return f"- Attempt ID: {self.attempt_id}\n- Retro Reactions:\n{reaction_strs}\n- Is Solved: {self.is_solved}\n- Unsolved Molecules:\n{unsolved_molecules_str}\n- Feedback:\n{feedback_str}"


class SynthesisContext(DataModel):
    """Storing fixed information that does not change during the synthesis planning.
    Should be used at tree level.
    """

    target_smiles: str
    user_constraint: Optional[str] = None
    previous_attempts: List[Union[PlanningAttempt, StrategicRoute]] = []
    strategy_overview: Optional[str] = None
    step_estimate: Optional[int] = None

    additional_notes: Optional[str] = None


class SynthesisPlanUpdate(DataModel):
    """Dynamic context of the synthesis. Should be used at node level"""

    target_smiles: str
    user_constraint: Optional[str] = None
    expandable_molecules: List[str] = []
    expandable_molecules_mapped: Optional[List[str]] = None
    mol_solve_states: Optional[List[bool]] = None
    previous_reactions: List[str] = []
    synthesis_context: Optional[SynthesisContext] = None
    strategy_overview: Optional[str] = None
    previous_steps: List[StrategicStep] = []
    next_steps: List[StrategicStep] = []
    next_retro_transformation: Optional[str] = None
    next_forward_reaction: Optional[str] = None
    expandable_molecule_index: Optional[int] = None
    reaction_atom_indices: Optional[list[int]] = None
    is_ring_break: Optional[bool] = None
    raw_response: Optional[str] = None
    model_used: Optional[str] = None

    @property
    def all_steps(self):
        """Combine previous and next steps into a single list."""
        all_steps = []
        if self.previous_steps:
            all_steps.extend(self.previous_steps)
        if self.next_steps:
            all_steps.extend(self.next_steps)
        return all_steps


class LLMGuidedNodeContext(DataModel):
    reaction_path: List[str]
    depth: int
    synthesis_context: Optional[Union[dict, SynthesisContext]] = None
    tree: Optional[Any] = None  # the tree class of MCTS

    def __post_init__(self):
        if self.synthesis_context is None and self.tree is not None:
            self.synthesis_context = getattr(self.tree, "_synthesis_plan", None)
        if isinstance(self.synthesis_context, dict):
            self.synthesis_context = SynthesisContext.from_dict(self.synthesis_plan)


@dataclass
class StrategyAlignmentScorerResponse:
    score: float
    justification: str
    raw_response: str
    prompt: str
