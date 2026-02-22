""" Module containing classes that implements different expansion policy strategies
"""

from __future__ import annotations

import abc
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from synthelite.agent_search.base import LLMBase
from synthelite.agent_search.schema import LLMGuidedNodeContext, SynthesisPlanUpdate
from synthelite.chem import SmilesBasedRetroReaction, TemplatedRetroReaction
from synthelite.chem.reaction import (
    AbsoluteRetroReaction,
    FixedRetroReaction,
    action_to_str,
    deduplicate_actions,
)
from synthelite.context.policy.utils import _make_fingerprint
from synthelite.query_embeddings.embeddings_search import FastTemplateSearchEngine
from synthelite.synthelite_utils.reaction_site import (
    get_action_changed_bond,
    is_equal_disconnection,
)
from synthelite.utils.exceptions import PolicyException
from synthelite.utils.clogging import logger
from synthelite.utils.models import load_model
from synthelite.synthelite_utils.transformer_request import TranslateReaxys

if TYPE_CHECKING:
    from synthelite.chem import TreeMolecule
    from synthelite.chem.reaction import RetroReaction
    from synthelite.context.config import Configuration
    from synthelite.utils.type_utils import (
        Any,
        Dict,
        List,
        Optional,
        Sequence,
        StrDict,
        Tuple,
    )
    from synthelite.search.mcts.node import LLMGuidedMctsNode


class ExpansionStrategy(abc.ABC):
    """
    A base class for all expansion strategies.

    The strategy can be used by either calling the `get_actions` method
    of by calling the instantiated class with a list of molecule.

    .. code-block::

        expander = MyExpansionStrategy("dummy", config)
        actions, priors = expander.get_actions(molecules)
        actions, priors = expander(molecules)

    :param key: the key or label
    :param config: the configuration of the tree search
    """

    _required_kwargs: List[str] = []
    _require_node_context: bool = False

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        if any(name not in kwargs for name in self._required_kwargs):
            raise PolicyException(
                f"A {self.__class__.__name__} class needs to be initiated "
                f"with keyword arguments: {', '.join(self._required_kwargs)}"
            )
        self._config = config
        self._logger = logger()
        self.key = key

    def __call__(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        return self.get_actions(molecules, cache_molecules)

    @abc.abstractmethod
    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
                                  policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """

    def reset_cache(self) -> None:
        """Reset the prediction cache"""


class ContextDependent:
    def set_node_context(self, search_graph_node: "LLMGuidedMctsNode"):
        """Set current node context for path-aware guidance for NextStepGenerator"""
        raise NotImplementedError("set_node_context must be implemented in subclass")


class MultiExpansionStrategy(ExpansionStrategy):
    """
    A base class for combining multiple expansion strategies.

    The strategy can be used by either calling the `get_actions` method
    or by calling the instantiated class with a list of molecules.

    :ivar expansion_strategy_keys: the keys of the selected expansion strategies
    :ivar additive_expansion: a conditional setting to specify whether all the actions
        and priors of the selected expansion strategies should be combined or not.
        Defaults to False.
    :ivar expansion_strategy_weights: a list of weights for each expansion strategy.
        The weights should sum to one. Exception is the default, where unity weight
        is associated to each strategy.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param expansion_strategies: the keys of the selected expansion strategies. All keys
        of the selected expansion strategies must exist in the expansion policies listed
        in config
    """

    _required_kwargs = ["expansion_strategies"]

    def __init__(
        self,
        key: str,
        config: Configuration,
        **kwargs: Any,
    ) -> None:
        super().__init__(key, config, **kwargs)
        self._config = config
        self._expansion_strategies: List[ExpansionStrategy] = []
        self.expansion_strategy_keys = kwargs["expansion_strategies"]

        self.cutoff_number = kwargs.get("cutoff_number")
        if self.cutoff_number:
            print(f"Setting multi-expansion cutoff_number: {self.cutoff_number}")

        self.expansion_strategy_weights = self._set_expansion_strategy_weights(kwargs)
        self.additive_expansion: bool = bool(kwargs.get("additive_expansion", False))
        self._logger.info(
            f"Multi-expansion strategy with policies: {self.expansion_strategy_keys}"
            f", and corresponding weights: {self.expansion_strategy_weights}"
        )

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies.

        The default implementation combines all the actions and priors of the
        selected expansion strategies into two lists respectively if the
        'additive_expansion' setting is set to True. This function can be overridden by
        a sub class to combine different expansion strategies in different ways.

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
            policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        expansion_strategies = self._get_expansion_strategies_from_config()

        all_possible_actions = []
        all_priors = []
        for expansion_strategy, expansion_strategy_weight in zip(
            expansion_strategies, self.expansion_strategy_weights
        ):
            possible_actions, priors = expansion_strategy.get_actions(
                molecules, cache_molecules
            )

            all_possible_actions.extend(possible_actions)
            if not self.additive_expansion and all_possible_actions:
                all_priors.extend(priors)
                break

            weighted_prior = [expansion_strategy_weight * p for p in priors]

            all_priors.extend(weighted_prior)

        all_possible_actions, all_priors = self._prune_actions(
            all_possible_actions, all_priors
        )
        return all_possible_actions, all_priors

    def _get_expansion_strategies_from_config(self) -> List[ExpansionStrategy]:
        if self._expansion_strategies:
            return self._expansion_strategies

        if not all(
            key in self._config.expansion_policy.items
            for key in self.expansion_strategy_keys
        ):
            raise ValueError(
                "The input expansion strategy keys must exist in the "
                "expansion policies listed in config"
            )
        self._expansion_strategies = [
            self._config.expansion_policy[key] for key in self.expansion_strategy_keys
        ]

        for expansion_strategy, weight in zip(
            self._expansion_strategies, self.expansion_strategy_weights
        ):
            if not getattr(expansion_strategy, "rescale_prior", True) and weight < 1:
                setattr(expansion_strategy, "rescale_prior", True)
                self._logger.info(
                    f"Enforcing {expansion_strategy.key}.rescale_prior=True"
                )
        return self._expansion_strategies

    def _prune_actions(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Prune the actions if a maximum number of actions is specified.

        :param actions: list of predicted actions
        :param priors: list of prediction probabilities
        :return: the top 'self.cutoff_number' actions and corresponding priors.
        """
        if not self.cutoff_number:
            return actions, priors

        sortidx = np.argsort(np.array(priors))[::-1].astype(int)
        priors = [priors[idx] for idx in sortidx[0 : self.cutoff_number]]
        actions = [actions[idx] for idx in sortidx[0 : self.cutoff_number]]
        return actions, priors

    def _set_expansion_strategy_weights(self, kwargs: StrDict) -> List[float]:
        """
        Set the weights of each expansion strategy using the input kwargs from config.
        The weights in the config should sum to one.
        If not set in the config file, the weights default to one for each strategy
        (for backwards compatibility).

        :param kwargs: input arguments to the MultiExpansionStrategy
        :raises: ValueError if weights from the config file do not sum to one.
        :return: a list of expansion strategy weights
        """
        if not "expansion_strategy_weights" in kwargs:
            return [1.0 for _ in self.expansion_strategy_keys]

        expansion_strategy_weights = kwargs["expansion_strategy_weights"]
        sum_weights = sum(expansion_strategy_weights)

        if sum_weights != 1:
            raise ValueError(
                "The expansion strategy weights in MultiExpansion should "
                "sum to one. -> "
                f"sum({expansion_strategy_weights})={sum_weights}."
            )

        return expansion_strategy_weights


class MultiExpansionStrategyFallback(MultiExpansionStrategy):
    """
    Sequentially applies multiple expansion strategies until one returns actions.
    Currently not compatible with method "get_actions_full_response"
    """

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        expansion_strategies = self._get_expansion_strategies_from_config()

        all_possible_actions = []
        all_priors = []
        for expansion_strategy, expansion_strategy_weight in zip(
            expansion_strategies, self.expansion_strategy_weights
        ):
            possible_actions, priors = expansion_strategy.get_actions(
                molecules, cache_molecules
            )
            if (
                possible_actions
            ):  # only move to the next strategy if no actions were found
                all_possible_actions = possible_actions
                weighted_prior = [expansion_strategy_weight * p for p in priors]
                all_priors = weighted_prior
                break

        all_possible_actions, all_priors = self._prune_actions(
            all_possible_actions, all_priors
        )
        return all_possible_actions, all_priors


class MultiExpansionPolicyLLMFallback(MultiExpansionStrategyFallback):
    """
    Expansion policy that tries LLM-guided expansion first; if no actions
    are returned, it falls back to a combined neural policy.

    - Primary: ``llm_guided`` (configurable via ``llm_policy_key``)
    - Fallback: ``multi_expansion_strategy`` (configurable via ``fallback_policy_key``)

    Returns the first non-empty set of actions. When using the LLM path that
    supports ``get_actions_full_response``, preserves the synthesis plan for
    downstream scoring/context.
    """

    def __init__(self, key: str, config: Configuration, **kwargs: Any) -> None:
        llm_policy_key = kwargs.pop("llm_policy_key", "llm_guided")
        fallback_policy_key = kwargs.pop(
            "fallback_policy_key", "multi_expansion_strategy"
        )

        expansion_keys = kwargs.get("expansion_strategies")
        if expansion_keys:
            # Ensure the configured keys include both llm and fallback, in the right order
            expansion_keys = [llm_policy_key, fallback_policy_key]
            kwargs["expansion_strategies"] = expansion_keys
        else:
            kwargs["expansion_strategies"] = [llm_policy_key, fallback_policy_key]

        super().__init__(key, config, **kwargs)

        # Store policy keys and enforce our ordering on the base class state
        self.llm_policy_key = llm_policy_key
        self.fallback_policy_key = fallback_policy_key
        self.expansion_strategy_keys = [self.llm_policy_key, self.fallback_policy_key]
        self._expansion_strategies = []  # reset cached strategies to respect new order

        self.ignore_stock = kwargs.get("ignore_stock", False)
        if hasattr(self.llm_policy, "ignore_stock"):
            self.llm_policy.ignore_stock = self.ignore_stock

    @property
    def llm_policy(self):
        return self._config.expansion_policy[self.llm_policy_key]

    @property
    def fallback_policy(self):
        return self._config.expansion_policy[self.fallback_policy_key]

    @property
    def next_step_generator(self):
        if hasattr(self.llm_policy, "next_step_generator"):
            return self.llm_policy.next_step_generator
        return None

    @property
    def do_verify_actions(self):
        return isinstance(
            self.llm_policy, DynamicLLMGuidedExpansionWithVerifierStrategy
        )

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        # Try LLM first
        llm_policy = self._config.expansion_policy[self.llm_policy_key]
        actions, priors = llm_policy.get_actions(molecules, cache_molecules)
        if actions:
            return self._prune_actions(actions, priors)
        logger.info(
            f"LLM policy {self.llm_policy_key} returned no actions, falling back to {self.fallback_policy_key}"
        )
        # Fallback to combined neural (e.g., USPTO + ringbreaker)
        fallback_policy = self._config.expansion_policy[self.fallback_policy_key]
        actions, priors = fallback_policy.get_actions(molecules, cache_molecules)
        return self._prune_actions(actions, priors)

    async def get_actions_full_response(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float], Any]:
        # Try LLM first, preserving synthesis plan if available
        llm_policy: LLMBasedExpansionStrategy = self._config.expansion_policy[
            self.llm_policy_key
        ]
        synthesis_plan = None
        if hasattr(llm_policy, "get_actions_full_response"):
            (
                actions,
                priors,
                synthesis_plan,
            ) = await llm_policy.get_actions_full_response(molecules, cache_molecules)
        else:
            actions, priors = llm_policy.get_actions(molecules, cache_molecules)
            synthesis_plan = None

        if actions:
            actions, priors = self._prune_actions(actions, priors)
            return actions, priors, synthesis_plan

        if synthesis_plan is None:  # stop signal hit
            return [], [], None

        # Fallback to combined neural (use LLM synthesis_plan for context if any)
        fallback_policy = self._config.expansion_policy[self.fallback_policy_key]
        if synthesis_plan.expandable_molecule_index is not None:
            molecules = [molecules[synthesis_plan.expandable_molecule_index]]
        # if synthesis_plan.is_ring_break:

        if hasattr(fallback_policy, "get_actions_full_response"):
            fb_actions, fb_priors, _ = await fallback_policy.get_actions_full_response(
                molecules, cache_molecules
            )
        else:
            fb_actions, fb_priors = fallback_policy.get_actions(
                molecules, cache_molecules
            )

        fb_actions, fb_priors = self._prune_actions(fb_actions, fb_priors)

        if fb_actions and self.do_verify_actions:
            fb_actions, fb_priors = await self.llm_policy.filter_actions(
                fb_actions, fb_priors, synthesis_plan, fallback_mode=True
            )

        for action in fb_actions:
            action.metadata[
                "policy_name"
            ] = self.key  # override policy name to combined

        return fb_actions, fb_priors, synthesis_plan


class TemplateBasedExpansionStrategy(ExpansionStrategy):
    """
    A template-based expansion strategy that will return `TemplatedRetroReaction` objects upon expansion.

    :ivar template_column: the column in the template file that contains the templates
    :ivar cutoff_cumulative: the accumulative probability of the suggested templates
    :ivar cutoff_number: the maximum number of templates to returned
    :ivar use_rdchiral: a boolean to apply templates with RDChiral
    :ivar use_remote_models: a boolean to connect to remote TensorFlow servers
    :ivar rescale_prior: a boolean to apply softmax to the priors
    :ivar chiral_fingerprints: if True will base expansion on chiral fingerprint
    :ivar mask: a boolean vector of masks for the reaction templates. The length of the vector should be equal to the
        number of templates. It is set to None if no mask file is provided as input.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param model: the source of the policy model
    :param template: the path to a HDF5 file with the templates
    :raises PolicyException: if the length of the model output vector is not same as the
        number of templates
    """

    _required_kwargs = [
        "model",
        "template",
    ]

    def __init__(self, key: str, config: Configuration, **kwargs: str) -> None:
        super().__init__(key, config, **kwargs)

        source = kwargs["model"]
        templatefile = kwargs["template"]
        maskfile: str = kwargs.get("mask", "")
        self.template_column: str = kwargs.get("template_column", "retro_template")
        self.cutoff_cumulative: float = float(kwargs.get("cutoff_cumulative", 0.995))
        self.cutoff_number: int = int(kwargs.get("cutoff_number", 50))
        self.use_rdchiral: bool = bool(kwargs.get("use_rdchiral", True))
        self.use_remote_models: bool = bool(kwargs.get("use_remote_models", False))
        self.rescale_prior: bool = bool(kwargs.get("rescale_prior", False))
        self.chiral_fingerprints = bool(kwargs.get("chiral_fingerprints", False))

        self._logger.info(
            f"Loading template-based expansion policy model from {source} to {self.key}"
        )
        self.model = load_model(source, self.key, self.use_remote_models)

        self._logger.info(f"Loading templates from {templatefile} to {self.key}")
        if templatefile.endswith(".csv.gz") or templatefile.endswith(".csv"):
            self.templates: pd.DataFrame = pd.read_csv(
                templatefile, index_col=0, sep="\t"
            )
        else:
            self.templates = pd.read_hdf(templatefile, "table")

        self.mask: Optional[np.ndarray] = (
            self._load_mask_file(maskfile) if maskfile else None
        )

        if hasattr(self.model, "output_size") and len(self.templates) != self.model.output_size:  # type: ignore
            raise PolicyException(
                f"The number of templates ({len(self.templates)}) does not agree with the "  # type: ignore
                f"output dimensions of the model ({self.model.output_size})"
            )
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
                                  policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """

        possible_actions = []
        priors: List[float] = []
        cache_molecules = cache_molecules or []
        self._update_cache(list(molecules) + list(cache_molecules))

        for mol in molecules:
            probable_transforms_idx, probs = self._cache[mol.inchi_key]
            possible_moves = self.templates.iloc[probable_transforms_idx]
            if self.rescale_prior:
                probs /= probs.sum()
            priors.extend(probs)
            for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                metadata = dict(move)
                del metadata[self.template_column]
                metadata["policy_probability"] = float(probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = move_index
                metadata["template"] = move[self.template_column]
                possible_actions.append(
                    TemplatedRetroReaction(
                        mol,
                        smarts=move[self.template_column],
                        metadata=metadata,
                        use_rdchiral=self.use_rdchiral,
                    )
                )
        return possible_actions, priors  # type: ignore

    def reset_cache(self) -> None:
        """Reset the prediction cache"""
        self._cache = {}

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        if self.mask is not None:
            predictions[~self.mask] = 0
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self.cutoff_cumulative):
            maxidx = int(np.argmin(cumsum < self.cutoff_cumulative))
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self.cutoff_number) or 1
        return sortidx[:maxidx]

    def _load_mask_file(self, maskfile: str) -> np.ndarray:
        self._logger.info(f"Loading masking of templates from {maskfile} to {self.key}")
        mask = np.load(maskfile)["arr_0"]
        if len(mask) != len(self.templates):
            raise PolicyException(
                f"The number of masks {len(mask)} does not match the number of templates {len(self.templates)}"
            )
        return mask

    def _update_cache(self, molecules: Sequence[TreeMolecule]) -> None:
        pred_inchis = []
        fp_list = []
        for molecule in molecules:
            if molecule.inchi_key in self._cache or molecule.inchi_key in pred_inchis:
                continue
            fp_list.append(
                _make_fingerprint(molecule, self.model, self.chiral_fingerprints)
            )
            pred_inchis.append(molecule.inchi_key)

        if not pred_inchis:
            return

        pred_list = np.asarray(self.model.predict(np.vstack(fp_list)))
        for pred, inchi in zip(pred_list, pred_inchis):
            probable_transforms_idx = self._cutoff_predictions(pred)
            self._cache[inchi] = (
                probable_transforms_idx,
                pred[probable_transforms_idx],
            )


class TemplateBasedDirectExpansionStrategy(TemplateBasedExpansionStrategy):
    """
    A template-based expansion strategy that will return `SmilesBasedRetroReaction` objects upon expansion
    by directly applying the template

    :param key: the key or label
    :param config: the configuration of the tree search
    :param source: the source of the policy model
    :param templatefile: the path to a HDF5 file with the templates
    :raises PolicyException: if the length of the model output vector is not same as the number of templates
    """

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
            policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """
        possible_actions = []
        priors = []

        super_actions, super_priors = super().get_actions(molecules, cache_molecules)
        for templated_action, prior in zip(super_actions, super_priors):
            for reactants in templated_action.reactants:
                reactants_str = ".".join(mol.smiles for mol in reactants)
                new_action = SmilesBasedRetroReaction(
                    templated_action.mol,
                    metadata=templated_action.metadata,
                    reactants_str=reactants_str,
                )
                possible_actions.append(new_action)
                priors.append(prior)

        return possible_actions, priors  # type: ignore


class ONMTExpansionStrategy(ExpansionStrategy):
    def __init__(
        self,
        key: str,
        # config: Configuration,
        **kwargs: Any,
    ) -> None:
        self.key = key
        # super().__init__(key, config, **kwargs)
        # self._config = config
        self._logger = logger()
        # dummy model
        self.model = TranslateReaxys()
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:

        possible_actions = []
        priors: List[float] = []
        cache_molecules = cache_molecules or []
        self._update_cache(list(molecules) + list(cache_molecules))

        for mol in molecules:
            moves_and_scores = self._cache[mol.inchi_key]

            for idx, tuple in enumerate(moves_and_scores):
                move, similarity = tuple

                metadata = {
                    "policy_probability": round(float(similarity), 4),
                    "policy_probability_rank": idx,
                    "policy_name": self.key,
                    "move": move,
                    "template_code": idx,
                    "template": "no_template",
                }

                possible_actions.append(
                    SmilesBasedRetroReaction(
                        mol,
                        metadata=metadata,
                        reactants_str=move,
                    )
                )

                priors.append(similarity)
        return possible_actions, priors

    def _prune_actions(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Prune the actions if a maximum number of actions is specified.

        :param actions: list of predicted actions
        :param priors: list of prediction probabilities
        :return: the top 'self.cutoff_number' actions and corresponding priors.
        """
        if not self.cutoff_number:
            return actions, priors

        sortidx = np.argsort(np.array(priors))[::-1].astype(int)
        priors = [priors[idx] for idx in sortidx[0 : self.cutoff_number]]
        actions = [actions[idx] for idx in sortidx[0 : self.cutoff_number]]
        return actions, priors

    def _update_cache(self, molecules: Sequence[TreeMolecule]) -> None:
        pred_inchis = []
        fp_list = []
        for molecule in molecules:
            if molecule.inchi_key in self._cache or molecule.inchi_key in pred_inchis:
                continue
            fp_list.append(molecule.smiles)
            pred_inchis.append(molecule.inchi_key)

        if not pred_inchis:
            return

        pred_list = np.asarray(self.model(fp_list))
        print(pred_list)
        for preds, inchi in zip(pred_list, pred_inchis):
            print(preds)
            self._cache[inchi] = [(pred["text"], pred["score"]) for pred in preds]


# class for LLM-guided expansion strategy as a top up on aiyznth
class LLMBasedExpansionStrategy(ExpansionStrategy, ContextDependent):
    def __init__(self, key: str, config, **kwargs):
        super().__init__(key, config, **kwargs)
        self._logger = logger()
        self._template_engine: Optional[FastTemplateSearchEngine] = None
        self._synthesis_plan = None

        self.ignore_stock = kwargs.get("ignore_stock", False)

        # Get LLM guidance config as dict
        llm_config = getattr(config.search, "llm_guidance", {})
        if not llm_config.get("enabled", False):
            raise ValueError("LLM guidance must be enabled in config")

        self.llm_config = llm_config
        self.config = config

        # store node context for LLM-guided expansion like path depth and reaction path in smiles format
        self._current_node_context = None
        self.use_rdchiral = kwargs.get("use_rdchiral", True)

    def _initialize_template_search(self):
        """Lazy initialization of LLM components"""
        # try:
        from synthelite.query_embeddings.embeddings_search import (
            FastTemplateSearchEngine,
        )

        # Access dict values
        if self.config.template_search is not None:
            self._template_engine = self.config.template_search
            self._logger.info(f"Using pre-configured template search engine")
        else:
            csv_path = self.llm_config["template_search"]["csv_path"]
            self._template_engine = FastTemplateSearchEngine(csv_path)
            self._logger.info(f"LLM-guided expansion initialized")

    def initialize_components(self):
        raise NotImplementedError(
            "initialize_components must be implemented in subclass"
        )

    def _generate_strategic_guidance(
        self, expandable_molecules: List[TreeMolecule]
    ) -> str:
        raise NotImplementedError("This method should be implemented in subclass")

    def _convert_to_actions(
        self,
        template_results: List[Dict],
        molecule: TreeMolecule,
        limits: Optional[int] = None,
        use_rdchiral: Optional[bool] = None,
    ) -> Tuple[List[TemplatedRetroReaction], List[float]]:
        """Convert template search results to actions with threshold filtering"""
        actions = []
        priors = []

        use_rdchiral = use_rdchiral if use_rdchiral is not None else self.use_rdchiral

        # GET THRESHOLD from config
        threshold = self.llm_config["template_search"]["similarity_threshold"]

        for i, result in enumerate(template_results.get("results", [])):
            similarity = result.get("similarity_score", 0.0)
            template_code = result.get("template_code", f"unknown_{i}")
            smarts = result.get("retro_template", "")

            # self._logger.debug(f"\n   Template {i}: {template_code}")
            # self._logger.debug(f"      Similarity: {similarity}")
            # self._logger.debug(f"      SMARTS: {smarts[:60]}...")

            try:
                action = TemplatedRetroReaction(
                    mol=molecule,
                    smarts=result["retro_template"],
                    metadata={
                        "policy_name": self.key,
                        "template_code": result.get("template_code", "unknown"),
                        "similarity_score": similarity,  #  Use actual similarity
                        "description": result.get("description", ""),
                        "use_rdchiral": use_rdchiral,
                        "lmdata": {
                            "lm_score": similarity
                            * 10,  # Convert to 0-10 scale, do not do that if you want to decrease influence of LLm, eg when trying to get more exploration?
                            "reasoning": f"Template selected based on {similarity:.3f} similarity to strategic guidance",
                            "query": template_results.get("query", ""),
                        },
                    },
                    # use_rdchiral=
                    use_rdchiral=use_rdchiral,  # RDChiral has issues with some templates
                )

                actions.append(action)
                priors.append(similarity)
                # if action.reactants and len(action.reactants) > 0:
                #     actions.append(action)
                #     priors.append(similarity)  # Use actual similarity as prior

                #     # Show successful reactants
                #     reactant_smiles = [mol.smiles for mol in action.reactants[0]]
                # self._logger.debug(f"      SUCCESS --> {reactant_smiles}")
                # else:
                # self._logger.debug(f"      NO REACTANTS")

            except Exception as e:
                print(f"      ERROR: {e}")
                continue

        # self._logger.info(f"\n   Final: {len(actions)} actions created from {len(template_results['results'])} templates (threshold: {threshold})")

        if limits is not None:
            actions = actions[:limits]
            priors = priors[:limits]

        return actions, priors

    def _convert_to_absolute_actions(
        self,
        actions: List[RetroReaction],
        priors: List[float],
        ref_atom_indices: Optional[List[int]] = None,
        ref_action: Optional[RetroReaction] = None,
        cutoff: int = 20,
    ) -> Tuple[List[AbsoluteRetroReaction], List[float]]:
        absolute_actions = []
        absolute_priors = []

        seen_reactant_strings = set()

        assert not (
            ref_atom_indices is not None and ref_action is not None
        ), "Only one of ref_atom_indices or ref_action can be provided"

        for action, prior in zip(actions, priors):
            for reactant_set in action.reactants:
                reactant_smi = ".".join(mol.smiles for mol in reactant_set)
                if reactant_smi in seen_reactant_strings:
                    continue
                seen_reactant_strings.add(reactant_smi)

                absolute_action = AbsoluteRetroReaction(
                    mol=action.mol,
                    reactants=(reactant_set,),
                    retro_action=action,
                    metadata=action.metadata,
                )
                if ref_atom_indices and self._validate_action_with_reference_mapping(
                    absolute_action, ref_atom_indices
                ):
                    absolute_actions.append(absolute_action)
                    absolute_priors.append(prior)

                elif ref_action and is_equal_disconnection(absolute_action, ref_action):
                    absolute_actions.append(absolute_action)
                    absolute_priors.append(prior)

            if len(absolute_actions) >= cutoff:
                break

        return absolute_actions, absolute_priors


class LLMGuidedExpansionStrategy(LLMBasedExpansionStrategy):
    """Expansion strategy using LLM guidance + embedding search"""

    def set_node_context(self, search_graph_node: "LLMGuidedMctsNode"):
        """Set current node context for path-aware guidance for NextStepGenerator"""
        node_context = search_graph_node.context_for_llm
        self._current_node_context = node_context
        # self._logger.debug(
        #     f"LLM Policy received node context: depth={node_context.get('depth', 0)}, path_length={len(node_context.get('reaction_path', []))}"
        # )

        # except Exception as e:
        #     self._logger.error(f"Failed to initialize LLM components: {e}")
        #     raise

    def initialize_components(self):
        try:
            self._initialize_template_search()
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM components: {e}")
            raise

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """Required abstract method implementation"""

        # Get strategy from tree via node context
        self._current_strategy = None
        if hasattr(self, "_current_node_context") and self._current_node_context:
            # _current_node_context is a dict with 'tree' reference
            if "tree" in self._current_node_context:
                tree = self._current_node_context["tree"]
                self._current_strategy = getattr(tree, "_synthesis_plan", None)
            else:
                # Fallback: Try to get tree from current expansion call context
                # We need to pass tree reference in the context dict
                pass

        if self._template_engine is None:
            self.initialize_components()

        if not molecules:
            return [], []

        actions = []
        priors = []

        for molecule in molecules:

            # Use NextStepGenerator for strategic guidance
            guidance: str = self._generate_strategic_guidance(molecule)

            # Get templates with strategic guidance
            try:
                if self._template_engine:
                    results = self._template_engine.search_templates(
                        query=guidance,
                        max_results=self.llm_config["template_search"]["max_templates"],
                        similarity_threshold=self.llm_config["template_search"][
                            "similarity_threshold"
                        ],
                    )
                    template_results = results.get("results", [])

                    self._logger.info(
                        f"Strategic search found {len(template_results)} templates"
                    )
                    self._logger.debug(
                        f"Engine found {len(template_results)} templates above threshold"
                    )

                else:
                    template_results = []

            except Exception as e:
                self._logger.warning(f"Template search failed: {e}")
                template_results = []

            # Convert to actions
            if template_results:
                mol_actions, mol_priors = self._convert_to_actions(
                    results,
                    molecule,
                    # limits=self.llm_config.llm_guidance.template_search.max_actions
                )
                actions.extend(mol_actions)
                priors.extend(mol_priors)
            else:
                self._logger.warning("No templates found for strategic guidance")

        actions, priors = deduplicate_actions(actions, priors)

        max_actions = self.llm_config["template_search"].get("max_actions", None)
        # max_actions = self.config.expansion.llm_guided.get('max_actions', None)

        if max_actions:
            # Limit actions if configured
            actions = actions[:max_actions]
            priors = priors[:max_actions]

        return actions, priors

    def _generate_strategic_guidance(self, molecule: TreeMolecule) -> str:
        """Generate strategic guidance using NextStepGenerator"""
        try:
            from synthelite.agent_search.next_step_generator import NextStepGenerator

            print(f"\n MOLECULE DEBUG:")
            print(f"   SMILES: {molecule.smiles}")
            print(
                f"   Available attributes: {[attr for attr in dir(molecule) if not attr.startswith('_')]}"
            )

            # Check if molecule has any path information
            if hasattr(molecule, "parent"):
                print(f"   Has parent: {molecule.parent}")
            if hasattr(molecule, "transform"):
                print(f"   Transform: {molecule.transform}")
            if hasattr(molecule, "_parent_node"):
                print(f"   Parent node: {getattr(molecule, '_parent_node', None)}")

            if self._current_node_context:
                reaction_path = self._current_node_context.get("reaction_path", [])
                current_depth = self._current_node_context.get("depth", 0)
                print(f"   Real context available:")
                print(f"      Depth: {current_depth}")
                print(f"      Path length: {len(reaction_path)}")
                if reaction_path:
                    print(
                        f"      Recent steps: {reaction_path[-2:] if len(reaction_path) >= 2 else reaction_path}"
                    )
            else:
                reaction_path = []
                current_depth = 0
                print(f"   No node context, using fallback")

            # Get synthesis strategy
            strategy = getattr(self, "_current_strategy", None)
            if not strategy:
                return f"retrosynthesis of {molecule.smiles}"

            # Create NextStepGenerator with model from config
            generator = NextStepGenerator(
                model=self.llm_config["next_step_generator"]["model"],
                temperature=self.llm_config["next_step_generator"]["temperature"],
            )

            # Generate strategic disconnection guidance
            strategic_guidance: str = generator.predict_next_step(
                current_smiles=molecule.smiles,
                reaction_path=reaction_path,
                current_depth=current_depth,
                strategy=strategy,
            )

            self._logger.info(f"NextStepGenerator guidance: {strategic_guidance}")
            self._logger.info(f" Strategic guidance for {molecule.smiles}")
            return strategic_guidance

        except Exception as e:
            self._logger.warning(f"Failed to generate strategic guidance: {e}")
            return f"retrosynthesis of {molecule.smiles}"

    def reset_cache(self):
        pass

    # def load_onmt(self, opt) -> Translator:
    #     self._logger.info(f"Loading ONMT model")
    #     return Translator.from_opt(opt)


class MolSetLLMGuidedExpansionStrategy(LLMBasedExpansionStrategy):
    def __init__(self, key: str, config: Configuration, **kwargs: Any) -> None:
        super().__init__(key, config, **kwargs)
        self._template_engine = None
        self._next_step_generator = None
        self.initialize_components()
        self._current_node_context: Optional[LLMGuidedNodeContext] = None
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def initialize_components(self):
        try:
            from synthelite.agent_search.next_step_generator import (
                NextStepGeneratorForBatch,
            )

            # from synthelite.query_embeddings.embeddings_search import FastTemplateSearchEngine

            # csv_path = self.llm_config['template_search']['csv_path']
            # self._template_engine = FastTemplateSearchEngine(csv_path)
            # self._logger.info(f"LLM-guided expansion initialized with templates from {csv_path}")
            self._initialize_template_search()

            self._next_step_generator = NextStepGeneratorForBatch(
                model=self.llm_config["next_step_generator"]["model"],
                temperature=self.llm_config["next_step_generator"]["temperature"],
            )
            self._logger.info(
                f"NextStepGenerator initialized with model {self.llm_config['next_step_generator']['model']}"
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM components: {e}")
            raise

    def set_node_context(self, search_graph_node: "LLMGuidedMctsNode"):
        """Set current node context for path-aware guidance for NextStepGenerator"""
        node_context = search_graph_node.context_for_llm
        self._current_node_context = LLMGuidedNodeContext.from_dict(node_context)
        # self._logger.debug(
        #     f"LLM Policy received node context: depth={node_context.get('depth', 0)}, path_length={len(node_context.get('reaction_path', []))}"
        # )

    @property
    def target_molecule(self):
        if not self._current_node_context:
            return None
        return self._current_node_context.target_molecule or None

    @property
    def synthesis_plan(self):
        if not self._current_node_context:
            return None
        return self._current_node_context.synthesis_plan.__dict__ or None

    @property
    def reaction_path(self):
        if not self._current_node_context:
            return None
        return getattr(self._current_node_context, "reaction_path", None)

    def _generate_strategic_guidance(
        self, expandable_molecules: List[TreeMolecule]
    ) -> str:
        target_molecule = self.target_molecule
        if not target_molecule:
            raise ValueError(
                "Target molecule is mandatory for NextStepGeneratorForBatch"
            )
        synthesis_plan = self.synthesis_plan
        if not synthesis_plan:
            raise ValueError("A strategy is mandatory for NextStepGeneratorForBatch")
        reaction_path = self.reaction_path
        if reaction_path is None:
            raise ValueError(
                "A reaction path must not be None for NextStepGeneratorForBatch"
            )

        try:
            strategic_guidance: str = self._next_step_generator.predict_next_step(
                target_molecule=target_molecule,
                current_smiles_batch=[each.smiles for each in expandable_molecules],
                reaction_path=reaction_path,
                strategy=synthesis_plan,
            )

            self._logger.info(f"NextStepGenerator guidance: {strategic_guidance.guide}")
            return strategic_guidance
        except Exception as e:
            raise e
            # self._logger.warning(f"Failed to generate strategic guidance: {e}")
            # return f"retrosynthesis of {target_molecule}"

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:

        strategic_guidance = self._generate_strategic_guidance(molecules).guide

        if "stop" in strategic_guidance.lower():
            self._logger.info("Strategic guidance indicates to stop expansion")
            return [], []

        retrieved_templates = self._template_engine.search_templates(
            query=strategic_guidance,
            max_results=self.llm_config["template_search"]["max_templates"],
            similarity_threshold=self.llm_config["template_search"][
                "similarity_threshold"
            ],
        )
        templates = retrieved_templates.get("results", [])
        self._logger.debug(f"Engine found {len(templates)} templates above threshold")

        actions = []
        priors = []

        if templates:
            for (
                molecule
            ) in (
                molecules
            ):  # TODO: Could be handle better by asking the LLM to output the molecule it wants to expand as well
                mol_actions, mol_priors = self._convert_to_actions(
                    retrieved_templates, molecule
                )
                actions.extend(mol_actions)
                priors.extend(mol_priors)
        else:
            self._logger.warning("No templates found for strategic guidance")

        actions, priors = deduplicate_actions(actions, priors)

        max_actions = self.llm_config["template_search"].get("max_actions", None)
        # max_actions = self.config.expansion.llm_guided.get('max_actions', None)

        if max_actions:
            # Limit actions if configured
            actions = actions[:max_actions]
            priors = priors[:max_actions]

        return actions, priors

        # TODO: implement caching of actions and priors


class DynamicLLMGuidedExpansionStrategy(LLMBasedExpansionStrategy):
    def __init__(self, key: str, config: Configuration, **kwargs: Any) -> None:
        super().__init__(key, config, **kwargs)
        self._template_engine = None
        self._next_step_generator = None
        self.initialize_components()
        self._current_node_context: Optional[LLMGuidedNodeContext] = None
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def initialize_components(self):
        try:
            from synthelite.agent_search.strategy_planner import (
                SynthesisStrategyPlanner,
            )

            # from synthelite.query_embeddings.embeddings_search import FastTemplateSearchEngine

            # csv_path = self.llm_config['template_search']['csv_path']
            # self._template_engine = FastTemplateSearchEngine(csv_path)
            # self._logger.info(f"LLM-guided expansion initialized with templates from {csv_path}")
            self._initialize_template_search()

            self._next_step_generator = SynthesisStrategyPlanner(
                model=self.llm_config["next_step_generator"]["model"],
                temperature=self.llm_config["next_step_generator"]["temperature"],
                backend=self.llm_config["next_step_generator"].get(
                    "backend", "openrouter"
                ),
            )
            self._logger.info(
                f"NextStepGenerator initialized with model {self.llm_config['next_step_generator']['model']}"
            )
        except Exception as e:
            self._logger.error(f"Failed to initialize LLM components: {e}")
            raise

    def set_node_context(self, search_graph_node: "LLMGuidedMctsNode"):
        """Set current node context for path-aware guidance for NextStepGenerator"""
        node_context = search_graph_node.context_for_llm
        self._current_node_context = (
            LLMGuidedNodeContext.from_dict(node_context)
            if isinstance(node_context, dict)
            else node_context
        )
        # self._logger.debug(
        #     f"LLM Policy received node context: {self._current_node_context.__dict__}"
        # )

    @property
    def target_molecule(self):
        if not self._current_node_context:
            return None
        return self._current_node_context.synthesis_context.target_smiles or None

    @property
    def synthesis_context(self):
        if not self._current_node_context:
            return None
        return self._current_node_context.synthesis_context or None

    @property
    def user_constraint(self):
        if not self._current_node_context:
            return None
        return self._current_node_context.synthesis_context.user_constraint

    @property
    def reaction_path(self):
        if not self._current_node_context:
            return None
        return getattr(self._current_node_context, "reaction_path", None)

    @property
    def previous_attempts(self):
        if not self._current_node_context:
            return []
        res = self._current_node_context.synthesis_context.previous_attempts or []
        return [each.to_dict() for each in res]

    async def _generate_strategic_guidance(
        self, expandable_molecules: List[TreeMolecule]
    ) -> str:
        user_constraint = self.user_constraint
        if not user_constraint:
            raise ValueError("user_constraint is mandatory ")
        target_molecule = self.target_molecule
        if not target_molecule:
            raise ValueError("target_molecule is mandatory ")
        reaction_path = self.reaction_path
        if reaction_path is None:
            raise ValueError("A reaction path must not be None")

        try:
            strategic_guidance: SynthesisPlanUpdate = (
                await self._next_step_generator.update_strategy_and_propose_next_step(
                    target_smiles=target_molecule,
                    expandable_molecules=[each.smiles for each in expandable_molecules],
                    previous_reactions=reaction_path,
                    user_constraint=user_constraint,
                )
            )

            self._logger.info(
                f"SynthesisStrategyPlanner next-step guidance: {strategic_guidance.next_forward_reaction}"
            )
            return strategic_guidance
        except Exception as e:
            raise e
            # self._logger.warning(f"Failed to generate strategic guidance: {e}")
            # return f"retrosynthesis of {target_molecule}"

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:

        actions, priors, _ = asyncio.run(
            self.get_actions_full_response(molecules, cache_molecules)
        )
        return actions, priors

        # TODO: implement caching of actions and priors

    async def get_actions_async(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:

        actions, priors, _ = await self.get_actions_full_response(
            molecules, cache_molecules
        )
        return actions, priors

        # TODO: implement caching of actions and priors

    async def get_actions_full_response(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ):
        synthesis_plan: SynthesisPlanUpdate = await self._generate_strategic_guidance(
            molecules
        )
        strategic_guidance = synthesis_plan.next_forward_reaction
        # if "stop" in strategic_guidance.lower():
        #     self._logger.info("Strategic guidance indicates to stop expansion")
        #     return [], []

        retrieved_templates = self._template_engine.search_templates(
            query=strategic_guidance,
            max_results=self.llm_config["template_search"]["max_templates"],
        )
        templates = retrieved_templates.get("results", [])
        self._logger.debug(f"Engine found {len(templates)} templates above threshold")

        actions = []
        priors = []

        if templates:
            for (
                molecule
            ) in (
                molecules
            ):  # TODO: Could be handle better by asking the LLM to output the molecule it wants to expand as well
                mol_actions, mol_priors = self._convert_to_actions(
                    retrieved_templates, molecule
                )
                actions.extend(mol_actions)
                priors.extend(mol_priors)
        else:
            self._logger.warning("No templates found for strategic guidance")

        actions, priors = deduplicate_actions(actions, priors)

        max_actions = self.llm_config["template_search"].get("max_actions", None)
        # max_actions = self.config.expansion.llm_guided.get('max_actions', None)

        if max_actions:
            # Limit actions if configured
            actions = actions[:max_actions]
            priors = priors[:max_actions]

        return actions, priors, synthesis_plan


class DynamicLLMGuidedExpansionWithVerifierStrategy(DynamicLLMGuidedExpansionStrategy):
    @property
    def max_actions(self):
        return self.llm_config["template_search"].get("max_actions", None)

    @property
    def reaction_selection_type(self):
        return self.llm_config["template_search"].get("selection_type", "verify")

    async def _generate_strategic_guidance(
        self, expandable_molecules: List[TreeMolecule]
    ) -> str:
        user_constraint = self.user_constraint
        if not user_constraint:
            raise ValueError("user_constraint is mandatory ")

        target_molecule = self.target_molecule
        if not target_molecule:
            raise ValueError("target_molecule is mandatory ")

        reaction_path = self.reaction_path
        if reaction_path is None:
            raise ValueError("A reaction path must not be None")

        previous_attempts = self.previous_attempts

        try:
            mol_solve_states = (
                [each in self.config.stock for each in expandable_molecules]
                if self.ignore_stock
                else None
            )

            strategic_guidance: Optional[
                SynthesisPlanUpdate
            ] = await self._next_step_generator.update_strategy_and_propose_next_step_with_search(
                target_smiles=target_molecule,
                previous_reactions=reaction_path,
                user_constraint=user_constraint,
                previous_attempts=previous_attempts,
                expandable_molecules=[each.smiles for each in expandable_molecules],
                expandable_molecules_mapped=[
                    each.mapped_smiles for each in expandable_molecules
                ],
                mol_solve_states=mol_solve_states,
            )

            if strategic_guidance is not None:
                self._logger.info(
                    f"SynthesisStrategyPlanner next-step guidance: {strategic_guidance.next_forward_reaction}"
                )

            return strategic_guidance
        except Exception as e:
            raise e

    def _validate_action_with_reference_mapping(
        self, action: RetroReaction, ref_mapping: list
    ):
        bond_edits = get_action_changed_bond(action)
        for bond_edit in bond_edits:
            for bond_map in bond_edit.bonds_lost_maps:
                if set(bond_map) & set(ref_mapping):
                    return True
            for bond_map in bond_edit.bonds_gained_maps:
                if set(bond_map) & set(ref_mapping):
                    return True
        return False

    def _filter_actions_with_ref_mapping(
        self, actions: list[RetroReaction], priors: list[float], ref_mapping: list
    ):
        filtered_actions = []
        filtered_priors = []
        for action, prior in zip(actions, priors):
            if self._validate_action_with_reference_mapping(action, ref_mapping):
                filtered_actions.append(action)
                filtered_priors.append(prior)
        return filtered_actions, filtered_priors

    async def get_actions_full_response(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ):
        cutoff = 20
        # return
        # actions, priors, synthesis_plan = await super().get_actions_full_response(molecules, cache_molecules)
        synthesis_plan: Optional[
            SynthesisPlanUpdate
        ] = await self._generate_strategic_guidance(molecules)
        if synthesis_plan is None:
            return [], [], None
        strategic_guidance = synthesis_plan.next_forward_reaction
        # if "stop" in strategic_guidance.lower():
        #     self._logger.info("Strategic guidance indicates to stop expansion")
        #     return [], []

        retrieved_templates = self._template_engine.search_templates(
            query=strategic_guidance,
            max_results=self.llm_config["template_search"]["max_templates"],
            similarity_threshold=self.llm_config["template_search"][
                "similarity_threshold"
            ],
        )
        templates = retrieved_templates.get("results", [])
        self._logger.debug(f"Engine found {len(templates)} templates above threshold")

        actions = []
        priors = []

        if synthesis_plan.expandable_molecule_index is not None:
            expand_molecules = [molecules[synthesis_plan.expandable_molecule_index]]
        else:
            expand_molecules = molecules

        if templates:
            for (
                molecule
            ) in (
                expand_molecules
            ):  # TODO: Could be handle better by asking the LLM to output the molecule it wants to expand as well
                mol_actions, mol_priors = self._convert_to_actions(
                    retrieved_templates,
                    molecule,
                    use_rdchiral=True
                    if synthesis_plan.is_ring_break
                    else self.use_rdchiral,
                )

                actions.extend(mol_actions)
                priors.extend(mol_priors)
        else:
            self._logger.warning("No templates found for strategic guidance")

        # actions, priors = deduplicate_actions(actions, priors)

        # seen_reactant_strings = set()

        # selected_actions = []
        # selected_priors = []

        # fixed_actions = []
        # fixed_priors = []

        # # n_llm_calls = 0

        # for action, prior in zip(actions, priors):
        #     # reaction_smis = action_to_str(action)
        #     for reactant_set in action.reactants:
        #         reactant_smi = ".".join(mol.smiles for mol in reactant_set)

        #         if reactant_smi in seen_reactant_strings:
        #             continue

        #         # reaction_smi = f"{reactant_smi}>>{action.mol.smiles}"
        #         # reaction_verdict = await self._next_step_generator.validate_found_reaction(synthesis_plan, reaction_smi)
        #         # n_llm_calls += 1

        #         seen_reactant_strings.add(reactant_smi)

        #         fixed_action = AbsoluteRetroReaction(
        #                 mol=action.mol,
        #                 reactants=(reactant_set, ),
        #                 retro_action=action,
        #                 metadata=action.metadata
        #             )

        #         if synthesis_plan.reaction_atom_indices and self._validate_action_with_reference_mapping(fixed_action, synthesis_plan.reaction_atom_indices):
        #             fixed_actions.append(fixed_action)
        #             fixed_priors.append(prior)

        #     if len(fixed_actions) >= cutoff:
        #         break
        fixed_actions, fixed_priors = self._convert_to_absolute_actions(
            actions, priors, synthesis_plan.reaction_atom_indices, cutoff=cutoff
        )

        self._logger.info(
            f"\n   Final: {len(fixed_actions)} actions created from {len(actions)} actions (threshold: {cutoff})"
        )

        fixed_actions = fixed_actions[:cutoff]
        fixed_priors = fixed_priors[:cutoff]

        selected_actions, selected_priors = await self.select_actions_with_llms(
            fixed_actions, fixed_priors, synthesis_plan, fallback_mode=False
        )
        # ref_reaction_mapping = synthesis_plan.reaction_atom_indices
        # if ref_reaction_mapping is not None:
        #     fixed_actions, fixed_priors = self._filter_actions_with_ref_mapping(fixed_actions, fixed_priors, ref_reaction_mapping)

        # fixed_actions = fixed_actions[:20]
        # fixed_priors = fixed_priors[:20]

        # selected_actions, selected_priors = await self.select_actions_with_llms(fixed_actions, fixed_priors, synthesis_plan)

        return selected_actions, selected_priors, synthesis_plan

    async def filter_actions(
        self,
        actions: List[RetroReaction],
        priors: List[float],
        synthesis_plan: Optional = None,
        fallback_mode: bool = False,
        cutoff: int = 20,
    ):
        if not isinstance(actions[0], AbsoluteRetroReaction):
            actions, priors = self._convert_to_absolute_actions(
                actions, priors, synthesis_plan.reaction_atom_indices, cutoff=cutoff
            )

        # ref_reaction_mapping = synthesis_plan.reaction_atom_indices if synthesis_plan else None
        # if ref_reaction_mapping is not None:
        #     actions, priors = self._filter_actions_with_ref_mapping(actions, priors, ref_reaction_mapping)

        actions = actions[:cutoff]
        priors = priors[:cutoff]

        selected_actions, selected_priors = await self.select_actions_with_llms(
            actions, priors, synthesis_plan, fallback_mode=fallback_mode
        )
        return selected_actions, selected_priors

    async def select_actions_with_llms(
        self,
        actions: List[RetroReaction],
        priors: List[float],
        synthesis_plan: Optional = None,
        fallback_mode: bool = False,
    ):
        if not actions:
            return [], []

        if not isinstance(actions[0], AbsoluteRetroReaction):
            actions, priors = self._convert_to_absolute_actions(actions, priors)

        if self.reaction_selection_type == "verify":
            (
                selected_actions,
                selected_priors,
            ) = await self._verify_found_reactions_with_llm(
                actions, priors, synthesis_plan
            )
        elif self.reaction_selection_type == "select":
            (
                selected_actions,
                selected_priors,
            ) = await self._select_found_reactions_with_llms(
                actions, priors, synthesis_plan, fallback_mode=fallback_mode
            )
        else:
            raise ValueError(
                f"Unknown reaction_selection_type: {self.reaction_selection_type}"
            )

        return selected_actions, selected_priors

    async def _verify_found_reactions_with_llm(
        self,
        reactions: List[AbsoluteRetroReaction],
        priors: List,
        synthesis_plan: Optional = None,
    ):
        n_llm_calls = 0
        selected_actions = []
        selected_priors = []

        for action, prior in zip(reactions, priors):
            reactant_smi = ".".join(mol.smiles for mol in action.reactants[0])
            reaction_smi = f"{reactant_smi}>>{action.mol.smiles}"

            reaction_verdict = await self._next_step_generator.validate_found_reaction(
                synthesis_plan, reaction_smi
            )
            n_llm_calls += 1

            if reaction_verdict:
                selected_actions.append(action)
                selected_priors.append(prior)

            if len(selected_actions) >= self.max_actions or n_llm_calls >= 10:
                break

        return selected_actions, selected_priors

    async def _select_found_reactions_with_llms(
        self,
        reactions: List[AbsoluteRetroReaction],
        priors: List,
        synthesis_plan: Optional = None,
        fallback_mode: bool = False,
    ):
        reaction_smis = []
        for action in reactions:
            reactant_smi = ".".join(mol.smiles for mol in action.reactants[0])
            reaction_smi = f"{reactant_smi}>>{action.mol.smiles}"
            reaction_smis.append(reaction_smi)

        selected_indices = await self._next_step_generator.select_found_reactions(
            synthesis_plan,
            reaction_smis,
            max_select=self.max_actions,
            fallback_mode=fallback_mode,
        )

        return [reactions[i] for i in selected_indices], [
            priors[i] for i in selected_indices
        ]


class LLMQueriedExpansionStrategy(ExpansionStrategy):
    "Explorative MCTS based on a list of natural language queries. Each query is one step."

    _required_kwargs = [
        # "queries",
        # "templates",
    ]
    _require_node_context = True

    def __init__(self, key: str, config: Configuration, **kwargs: Any) -> None:
        super().__init__(key, config, **kwargs)

        if "templates" in kwargs:
            self.search_engine = FastTemplateSearchEngine(
                csv_with_embeddings=kwargs["templates"]
            )
        else:
            self.search_engine = config.template_search
            if self.search_engine is None:
                raise ValueError(
                    "Either templates are provided in kwargs or initialized in config.template_search"
                )

        self.alpha = kwargs.get("alpha", 0.5)

        self._queries = kwargs.get("queries", [])
        self._ref_actions = kwargs.get("ref_actions", [])

        self.templates = self.search_engine.df
        self.template_column = kwargs.get("template_column", "retro_template")
        self.template_code_column = kwargs.get("template_code_column", "template_code")

        self.templates = self.templates.set_index(self.template_code_column)

        self._logger.info(f"Loading LLM-queried expansion policy to {self.key}")
        self._logger.info(
            f"Loading templates from {self.search_engine.templatefile} to {self.key}"
        )

        self.cutoff_search_similarity = float(
            kwargs.get("cutoff_search_similarity", 0.9)
        )
        self.cutoff_search_templates = int(kwargs.get("cutoff_search_templates", 500))

        self.cutoff_number = int(kwargs.get("cutoff_number", 50))

        self.use_rdchiral = bool(kwargs.get("use_rdchiral", True))
        self.rescale_prior = bool(kwargs.get("rescale_prior", False))

        self.ignore_stock = bool(kwargs.get("ignore_stock", False))
        # if "model" in kwargs:
        #     self.use_remote_models: bool = bool(kwargs.get("use_remote_models", False))
        #     self.model = load_model(kwargs["model"], self.key, use_remote=False)
        #     self.chiral_fingerprints = bool(kwargs.get("chiral_fingerprints", False))
        # else:
        #     self.model = None
        #     self.use_remote_models = False
        #     self.chiral_fingerprints = False

        self._current_step = 0
        self._current_step_updated = False

        self._cache = {}

        self._template_search_cache = defaultdict(list)

    @property
    def queries(self):
        return self._queries

    @queries.setter
    def queries(self, queries: List[str]):
        self._queries = queries
        self.reset_cache()

    @property
    def ref_actions(self):
        return self._ref_actions

    @ref_actions.setter
    def ref_actions(self, actions: List[RetroReaction]):
        self._ref_actions = actions
        self.reset_cache()

    def set_ref_queries_and_actions(
        self, queries: List[str], actions: List[RetroReaction]
    ):
        assert len(queries) == len(
            actions
        ), "Number of queries must match number of reference actions"
        self._queries = queries
        self._ref_actions = actions
        self.reset_cache()

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, step: int):
        self._current_step_updated = True
        self._current_step = step

    def reset_cache(self):
        self._cache = {}
        self._template_search_cache = defaultdict(list)

    def _get_mol_key(self, molecule: TreeMolecule) -> str:
        return f"{molecule.inchi_key}_step{self.current_step}"

    def set_node_context(self, search_graph_node: "LLMGuidedMctsNode"):
        current_step = len(search_graph_node.path_to()[0])  # current step is zero-based
        self.current_step = current_step
        self._logger.debug(
            f"LLMQueriedExpansionStrategy set current_step to {self.current_step}"
        )

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:

        if self.queries and self.ref_actions:
            assert len(self.queries) == len(
                self.ref_actions
            ), "Number of queries must match number of reference actions"

        if not self._current_step_updated:
            self._logger.warning(
                "current_step was not updated before expansion. Make sure to set it correctly before every call to get_actions."
            )

        # ref_action = self.ref_actions[self.current_step] if 0 <= self.current_step < len(self.ref_actions) else None

        possible_actions = []
        priors: List[float] = []
        cache_molecules = cache_molecules or []
        self._update_cache(list(molecules) + list(cache_molecules))

        for mol in molecules:
            mol_key = self._get_mol_key(mol)
            cached_actions, probs = self._cache.get(mol_key, ([], np.array([])))
            actions = []
            for cached_action in cached_actions:
                # Create a copy of the action so that the molecule object is not shared between actions
                actions.append(
                    AbsoluteRetroReaction(
                        mol=mol,
                        # reactants=cached_action.reactants,
                        reactants_mapped_smiles=cached_action.reactants_mapped_smiles,
                        index=cached_action.index,
                        metadata=cached_action.metadata,
                        smarts=cached_action.smarts,
                    )
                )
            #     probable_transforms_idx, probs = self._cache.get(mol_key, (np.array([]), np.array([])))
            #     possible_moves = self.templates.loc[probable_transforms_idx]
            if probs.size > 0 and self.rescale_prior:
                probs /= probs.sum()

            possible_actions.extend(actions)
            priors.extend(probs.tolist())
        #     # priors.extend(probs)
        #     for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
        #         metadata = dict()
        #         # del metadata[self.template_column]
        #         metadata["template_hash"] = move.get("template_hash", "")
        #         metadata["library_occurence"] = int(move.get("library_occurence", -1))
        #         metadata["llm_description"] = move.get("llm_description", "")
        #         metadata["query_similarity"] = float(probs[idx].round(4))
        #         # metadata["policy_probability_rank"] = idx
        #         metadata["policy_name"] = self.key
        #         metadata["template_code"] = move_index
        #         metadata["template"] = move[self.template_column]
        #         metadata["query"] = self.queries[self.current_step] if 0 <= self.current_step < len(self.queries) else ""

        #         templated_actions = TemplatedRetroReaction(
        #             mol,
        #             smarts=move[self.template_column],
        #             metadata=metadata,
        #             use_rdchiral=self.use_rdchiral,
        #         )

        #         abs_actions = templated_actions.to_absolute_reactions()

        #         for action in abs_actions:
        #             if is_equal_disconnection(action, ref_action):
        #                 possible_actions.append(action)
        #                 priors.append(probs[idx])
        #         # possible_actions.append(
        #         #     TemplatedRetroReaction(
        #         #         mol,
        #         #         smarts=move[self.template_column],
        #         #         metadata=metadata,
        #         #         use_rdchiral=self.use_rdchiral,
        #         #     )
        #         # )
        # possible_actions = possible_actions[:self.cutoff_number]
        # priors = priors[:self.cutoff_number]
        return possible_actions, priors

    # def _update_cache_with_neural_model(self, molecules: Sequence[TreeMolecule]) -> None:
    #     return super()._update_cache(molecules)

    def _update_cache(self, molecules: Sequence[TreeMolecule]) -> None:
        mol_keys = [self._get_mol_key(mol) for mol in molecules]

        if all(mol_key in self._cache for mol_key in mol_keys):
            return

        current_step = self.current_step
        query = (
            self.queries[current_step]
            if 0 <= current_step < len(self.queries)
            else None
        )

        ref_action = (
            self.ref_actions[self.current_step]
            if 0 <= self.current_step < len(self.ref_actions)
            else None
        )

        # ref_action = self.ref_actions[current_step] if 0 <= current_step < len(self.ref_actions) else None
        # if query is None and self.model is not None:
        #     # self._update_cache_with_neural_model(molecules)
        #     return
        if query is None:
            self._logger.warning(
                f"No query found for current_step {current_step}. Skipping template search."
            )
            return

        if current_step not in self._template_search_cache:
            # search results only depend on the current step.
            search_res = self.search_engine.search_templates(
                query=query,
                max_results=self.cutoff_search_templates,
                similarity_threshold=self.cutoff_search_similarity,
                alpha=self.alpha,
            )
            search_results = search_res.get("results", [])
            search_from_cache = False
        else:
            search_results = self._template_search_cache[current_step]
            search_from_cache = True
            # search_results = search_res.get('results', [])

        self._logger.debug(
            f"Found {len(search_results)} templates for query '{query}' at step {current_step} above threshold similarity {self.cutoff_search_similarity}"
        )

        seen_actions = set()
        for molecule in molecules:
            # template_ids = []
            # template_similarities = []
            possible_actions = []
            possible_probs = []

            mol_key = self._get_mol_key(molecule)
            if mol_key in self._cache:
                continue

            for search_result in search_results:
                # Early termination if we already have enough actions
                if len(possible_actions) >= self.cutoff_number:
                    break

                template_code = int(search_result["template_code"])
                move = self.templates.loc[template_code]
                prob = search_result["search_score"]
                metadata = dict()
                # del metadata[self.template_column]
                metadata["template_hash"] = move.get("template_hash", "")
                metadata["library_occurence"] = int(move.get("library_occurence", -1))
                metadata["llm_description"] = move.get("llm_description", "")
                metadata["query_similarity"] = round(prob, 4)
                # metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = search_result["template_code"]
                metadata["template"] = move[self.template_column]
                metadata["query"] = (
                    self.queries[self.current_step]
                    if 0 <= self.current_step < len(self.queries)
                    else ""
                )

                templated_action = TemplatedRetroReaction(
                    mol=molecule,
                    smarts=search_result["retro_template"],
                    metadata=metadata,
                    use_rdchiral=ref_action.metadata.get(
                        "use_rdchiral", self.use_rdchiral
                    ),
                )

                if not templated_action.reactants:
                    continue

                abs_actions = templated_action.to_absolute_reactions()

                for action in abs_actions:
                    action_str = tuple(action_to_str(action, canonicalize=True))
                    if action_str in seen_actions:
                        continue
                    seen_actions.add(action_str)
                    try:
                        if is_equal_disconnection(action, ref_action):
                            possible_actions.append(action)
                            possible_probs.append(prob)

                            if not search_from_cache:
                                self._template_search_cache[current_step].append(
                                    search_result
                                )

                            # Early termination if we have enough actions
                            if len(possible_actions) >= self.cutoff_number:
                                break
                    except Exception as e:
                        print(f"Error with reaction: {action_str}")
                        raise e

                # Early termination if we have enough actions
                if len(possible_actions) >= self.cutoff_number:
                    break

                # template_ids.append(int(search_result['template_code']))
                # template_similarities.append(float(search_result['search_score']))

            possible_actions = possible_actions[: self.cutoff_number]
            possible_probs = possible_probs[: self.cutoff_number]
            possible_probs = np.array(possible_probs)
            # template_ids = np.array(template_ids)
            # template_similarities = np.array(template_similarities)
            # template_ids = template_ids[:self.cutoff_number]
            # template_similarities = template_similarities[:self.cutoff_number]

            self._cache[mol_key] = (possible_actions, possible_probs)
