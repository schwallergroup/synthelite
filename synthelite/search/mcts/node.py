""" Module containing a class that represents a node in the search tree.
"""
from __future__ import annotations

import asyncio
import random
import json
import re
from typing import TYPE_CHECKING, Dict, List, Optional
import asyncio
import json
import litellm
import numpy as np
import sys
import os

from time import sleep
from paretoset import paretoset

from synthelite.agent_search.schema import (
    LLMGuidedNodeContext,
    StrategyAlignmentScorerResponse,
    SynthesisContext,
    SynthesisPlanUpdate,
)
from synthelite.chem import TreeMolecule, deserialize_action, serialize_action
from synthelite.search.mcts.state import MctsState
from synthelite.search.mcts.utils import (
    ReactionTreeFromSuperNode,
    normalize_heuristic_scores,
    route_to_node,
)
from synthelite.search.mcts.heuristic import HEURISTIC_SCORER
from synthelite.utils.exceptions import (
    NodeUnexpectedBehaviourException,
    RejectionException,
)
from typing import Any
from synthelite.utils.clogging import logger
from synthelite.synthelite_utils import check_template
from rxnutils.chem.reaction import ChemicalReaction, ReactionException

if TYPE_CHECKING:
    from synthelite.chem import (
        MoleculeDeserializer,
        MoleculeSerializer,
        RetroReaction,
    )
    from synthelite.context.config import Configuration
    from synthelite.reactiontree import ReactionTree
    from synthelite.search.mcts.search import MctsSearchTree
    from synthelite.utils.type_utils import List, Optional, StrDict, Tuple
    from synthelite.context.policy.expansion_strategies import (
        LLMGuidedExpansionStrategy,
    )

from rdkit import Chem

import nest_asyncio

nest_asyncio.apply()


def canon_mol(mol):
    return Chem.MolToSmiles(Chem.MolFromSmiles(mol))


class MctsNode:
    """
    A node in the search tree.

    The children are instantiated lazily for efficiency: only when
    a child is selected the reaction to create that child is applied.

    Properties of an instantiated children to a node can be access with:

    .. code-block::

        children_attr = node[child]

    the return value is a dictionary with keys "action", "value", "prior"
    and "visitations".

    :ivar is_expanded: if the node has had children added to it
    :ivar is_expandable: if the node is expandable
    :ivar tree: the tree owning this node

    :param state: the state of the node
    :param owner: the tree that owns this node
    :param config: settings of the tree search algorithm
    :param parent: the parent node, defaults to None
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[MctsNode] = None,
    ):
        self._state = state
        self._config = config
        self._expansion_policy = config.expansion_policy
        self._filter_policy = config.filter_policy
        self.tree = owner
        self.is_expanded: bool = False
        self.is_expandable: bool = not self.state.is_terminal
        self._parent = parent

        if owner is None:
            self.created_at_iteration: Optional[int] = None
        else:
            self.created_at_iteration = self.tree.profiling["iterations"]

        self._children_values: List[float] = []
        self._children_priors: List[float] = []
        self._children_visitations: List[int] = []
        self._children_actions: List[RetroReaction] = []
        self._children: List[Optional[MctsNode]] = []

        self.blacklist = set(mol.inchi_key for mol in state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

        if self._algo_config["mcts_grouping"]:
            self._degeneracy_check = self._algo_config["mcts_grouping"].lower()
        else:
            self._degeneracy_check = "none"
        self._logger = logger()

    def __getitem__(self, node: "MctsNode") -> StrDict:
        idx = self._children.index(node)
        return {
            "action": self._children_actions[idx],
            "value": self._children_values[idx],
            "prior": self._children_priors[idx],
            "visitations": self._children_visitations[idx],
        }

    @classmethod
    def create_root(
        cls, smiles: str, tree: MctsSearchTree, config: Configuration
    ) -> "MctsNode":
        """
        Create a root node for a tree using a SMILES.

        :param smiles: the SMILES representation of the root state
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :return: the created node
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = MctsState(mols=[mol], config=config)
        return cls(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        tree: MctsSearchTree,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: Optional["MctsNode"] = None,
    ) -> "MctsNode":
        """
        Create a new node from a dictionary, i.e. deserialization.

        :param dict_: the serialized node
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :param molecules: the deserialized molecules
        :param parent: the parent node
        :return: a deserialized node
        """
        # pylint: disable=protected-access
        state = MctsState.from_dict(dict_["state"], config, molecules)
        node = cls(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children_actions = [
            deserialize_action(action_dict, molecules)
            for action_dict in dict_["children_actions"]
        ]
        node._children = [
            cls.from_dict(child, tree, config, molecules, parent=node)
            if child
            else None
            for child in dict_["children"]
        ]
        return node

    @property
    def children(self) -> List["MctsNode"]:
        """
        Returns all of the instantiated children.

        :return: the children
        """
        return [child for child in self._children if child]

    @property
    def is_solved(self) -> bool:
        """Return if the state is solved."""
        return self.state.is_solved

    @property
    def parent(self) -> Optional["MctsNode"]:
        """Return the parent of the node."""
        return self._parent

    @property
    def state(self) -> MctsState:
        """Return the underlying state of the node."""
        return self._state

    @property
    def _algo_config(self) -> StrDict:
        """Just a convinient, shorter name of this."""
        return self._config.search.algorithm_config

    @property
    def ignore_stock(self):
        return getattr(self.config.expansion_policy, "ignore_stock", False)

    def actions_to(self) -> List[RetroReaction]:
        """
        Returns the actions leading to this node

        :return: the list of actions
        """
        return self.path_to()[0]

    def backpropagate(self, child: "MctsNode", value_estimate: float) -> None:
        """
        Update the number of visitations of a particular child and its value.

        :param child: the child node
        :param value_estimate: the value to add to the child value
        """
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        self._children_values[idx] += value_estimate

    def children_view(self) -> StrDict:
        """
        Creates a view of the children attributes. Each of the
        list returned is a new list, although the actual children
        are not copied.

        The return dictionary will have keys "actions", "values",
        "priors", "visitations" and "objects".

        :return: the view
        """
        return {
            "actions": list(self._children_actions),
            "values": list(self._children_values),
            "priors": list(self._children_priors),
            "visitations": list(self._children_visitations),
            "objects": list(self._children),
        }

    def expand(self) -> None:
        """
        Expand the node.

        Expansion is the process of creating the children of the node,
        without instantiating a child object. The actions and priors are
        taken from the policy network.

        If immediate instantiation is marked for some policies, however, the
        children nodes will be instantiated.
        """
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            self._logger.debug(msg)
            raise NodeUnexpectedBehaviourException(msg)

        if self.is_expanded or not self.is_expandable:
            return

        self.is_expanded = True

        cache_molecules = []
        if self.parent:
            for child in self.parent.children:
                if child is not self:
                    cache_molecules.extend(child.state.expandable_mols)

        # Calculate the possible actions, fill the child_info lists
        # Actions by default only assumes 1 set of reactants
        actions, priors = self._expansion_policy(
            self.state.expandable_mols, cache_molecules
        )
        self._fill_children_lists(actions, priors)

        # Reverse the expansion if it did not produce any children
        if len(actions) == 0:
            self.is_expandable = False
            self.is_expanded = False

        if self.tree:
            self.tree.profiling["expansion_calls"] += 1

        if not self._algo_config["immediate_instantiation"]:
            return
        # Instantiate all children actions created by the marked policy,
        # a new list of actions will be iterated over, because it can grow due
        # to instantiation
        nactions = len(actions)
        for child_idx, action in enumerate(self._children_actions[:nactions]):
            policy_name = action.metadata.get("policy_name")
            if (
                policy_name
                and policy_name in self._algo_config["immediate_instantiation"]
            ):
                self._instantiate_child(child_idx)

    def is_terminal(self) -> bool:
        """
        Node is terminal if its unexpandable, or the internal state is terminal (solved).

        :return: the terminal attribute of the node
        """
        # return not self.is_expandable or (self.state.is_terminal and not getattr(self.config, "ignore_stock", False))
        return (
            self.state.max_transforms >= self.config.search.max_transforms
            or (self.state.is_solved and not self.ignore_stock)
            or not self.is_expandable
        )

    def path_to(self, to_node=None) -> Tuple[List[RetroReaction], List[MctsNode]]:
        """
        Return the path to this node, which is a list of actions and a list of node.

        :return: the actions and nodes
        """
        return route_to_node(self, to_node=to_node)

    def promising_child(self) -> Optional["MctsNode"]:
        """
        Return the child with the currently highest Q+U.

        The selected child will be instantiated if it has not been already.

        If no actions could be found that were applicable, the method will
        return None.

        :return: the child
        """
        child = None
        while child is None:
            try:
                child = self._score_and_select()
            # _score_and_select raises exception if no children can be selected
            except ValueError:
                child = None
                break

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            self.is_expanded = False
            self.is_expandable = False

        return child

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the node object to a dictionary.

        :param molecule_store: the serialized molecules
        :return: the serialized node
        """
        return {
            "state": self.state.serialize(molecule_store),
            "children_values": self._serialize_stats_list("_children_values"),
            "children_priors": self._serialize_stats_list("_children_priors"),
            "children_visitations": self._children_visitations,
            "children_actions": [
                serialize_action(action, molecule_store)
                for action in self._children_actions
            ],
            "children": [
                child.serialize(molecule_store) if child else None
                for child in self._children
            ],
            "is_expanded": self.is_expanded,
            "is_expandable": self.is_expandable,
        }

    def to_reaction_tree(self) -> ReactionTree:
        """
        Return reaction tree from the path of actions and nodes leading to this node.

        :return: the constructed tree
        """
        return ReactionTreeFromSuperNode(self).tree

    def _check_child_reaction(self, reaction: RetroReaction) -> bool:
        if not reaction.reactants:
            self._logger.debug(f"{reaction} did not produce any reactants")
            return False

        # fmt: off
        reactants0 = reaction.reactants[0]
        if len(reaction.reactants) == 1 and len(reactants0) == 1 and reaction.mol == reactants0[0]:
            return False
        # fmt: on

        return True

    def _children_q(self) -> np.ndarray:
        return np.array(self._children_values) / np.array(self._children_visitations)

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._algo_config["C"] * np.sqrt(2 * total_visits / child_visits)

    def _create_children_nodes(
        self, states: List[MctsState], child_idx: int
    ) -> List["MctsNode"]:
        new_nodes = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            if self._generated_degeneracy(state, first_child_idx):
                # Only need to disable first new child,
                # if the action generated more states, we will just not generate
                # a child for that state
                if state_index == 0:
                    self._disable_child(child_idx)
                continue

            # If there's more than one outcome, the lists need be expanded
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            if self._filter_child_reaction(self._children_actions[child_idx]):
                self._disable_child(child_idx)
            else:
                new_node = self.__class__(
                    state=state, owner=self.tree, config=self._config, parent=self
                )
                self._children[child_idx] = new_node
                new_nodes.append(new_node)
        return new_nodes

    def _disable_child(self, child_idx: int) -> None:
        self._children_values[child_idx] = -1e6

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        new_action = self._children_actions[old_index].copy(index=action_index)
        self._children_actions.append(new_action)
        self._children_priors.append(self._children_priors[old_index])
        self._children_values.append(self._children_values[old_index])
        self._children_visitations.append(self._children_visitations[old_index])
        self._children.append(None)
        return len(self._children) - 1

    def _fill_children_lists(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> None:
        self._children_actions = actions
        self._children_priors = priors
        nactions = len(actions)
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        if self._algo_config["use_prior"]:
            self._children_values = list(self._children_priors)
        else:
            self._children_values = [self._algo_config["default_prior"]] * nactions

    def _filter_child_reaction(self, reaction: RetroReaction) -> bool:
        if self._regenerated_blacklisted(reaction):
            self._logger.debug(
                f"Reaction {reaction.reaction_smiles()} "
                f"was rejected because it re-generated molecule not in stock"
            )
            return True

        if not self._filter_policy.selection:
            return False
        try:
            self._filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _generated_degeneracy(self, new_state: MctsState, child_idx: int) -> bool:
        """
        Check if a new MCTS state is equal to another MCTS state of a children node.

        The check can be "partial" in which the equality is based only on the expandable molecules,
        or "full" in which the equality is based on all molecules in the state.

        The comparison will not be made on unexpanded children nodes
        or terminal children nodes.

        The metadata of the degenerate action will be added to the metadata
        of the previously created equal state.
        """

        def equal_states(query_state):
            if self._degeneracy_check == "partial":
                return query_state.expandables_hash == new_state.expandables_hash
            return query_state == new_state

        if self._degeneracy_check not in ["partial", "full"]:
            return False
        previous_action = None
        for child, action in zip(self._children, self._children_actions):
            if (
                child is not None
                and not child.is_terminal()
                and equal_states(child.state)
            ):
                previous_action = action
                break

        if previous_action is None:
            return False

        # No need to copy the metadata because it will be the same
        if previous_action is self._children_actions[child_idx]:
            return True

        metadata_copy = dict(self._children_actions[child_idx].metadata)
        if "additional_actions" not in previous_action.metadata:
            previous_action.metadata["additional_actions"] = []
        previous_action.metadata["additional_actions"].append(metadata_copy)
        return True

    def _instantiate_child(self, child_idx: int) -> List["MctsNode"]:
        """
        Instantiate the children node.

        The algorithm is:
        * Apply the reaction associated with the child
        * If the application of the action failed, set value to -1e6 and return None
        * Create a new state array, one new state for each of the reaction outcomes
        * Create new child nodes
            - If a filter policy is available and the reaction outcome is unlikely
              set value of child to -1e6
         * Return all new nodes
        """
        if self._children[child_idx] is not None:
            raise NodeUnexpectedBehaviourException("Node already instantiated")

        reaction = self._children_actions[child_idx]
        if reaction.unqueried:
            if self.tree:
                self.tree.profiling["reactants_generations"] += 1
            _ = reaction.reactants

        if not self._check_child_reaction(reaction):
            self._disable_child(child_idx)
            return []

        keep_mols = [mol for mol in self.state.mols if mol is not reaction.mol]
        new_states = [
            MctsState(keep_mols + list(reactants), self._config)
            for reactants in reaction.reactants
        ]
        return self._create_children_nodes(new_states, child_idx)

    def _regenerated_blacklisted(self, reaction: RetroReaction) -> bool:
        if not self._algo_config["prune_cycles_in_search"]:
            return False
        for reactants in reaction.reactants:
            for mol in reactants:
                if mol.inchi_key in self.blacklist:
                    return True
        return False

    def _score_and_select(self) -> Optional["MctsNode"]:
        if not max(self._children_values) > 0:
            raise ValueError("Has no selectable children")
        scores = self._children_q() + self._children_u()
        indices = np.where(scores == scores.max())[0]
        index = np.random.choice(indices)
        return self._select_child(index)

    def _select_child(self, child_idx: int) -> Optional["MctsNode"]:
        """
        Selecting a child node implies instantiating the children nodes.

        If the child has already been instantiated, return immediately
        Otherwise, select a random node of the feasible ones to return
        """
        if self._children[child_idx]:
            return self._children[child_idx]

        new_nodes = self._instantiate_child(child_idx)
        if new_nodes:
            return random.choice(new_nodes)
        return None

    def _serialize_stats_list(self, name: str) -> List[float]:
        return [float(value) for value in getattr(self, name)]

    def reset_expansion(self):
        self.is_expanded = False
        self.is_expandable = not self.state.is_terminal
        self._children_values = []
        self._children_priors = []
        self._children_visitations = []
        self._children_actions = []
        self._children = []


class LLMGuidedMctsNode(MctsNode):
    """
    A node in the search tree.

    The children are instantiated lazily for efficiency: only when
    a child is selected the reaction to create that child is applied.

    Properties of an instantiated children to a node can be access with:

    .. code-block::

        children_attr = node[child]

    the return value is a dictionary with keys "action", "value", "prior"
    and "visitations".

    :ivar is_expanded: if the node has had children added to it
    :ivar is_expandable: if the node is expandable
    :ivar tree: the tree owning this node

    :param state: the state of the node
    :param owner: the tree that owns this node
    :param config: settings of the tree search algorithm
    :param parent: the parent node, defaults to None
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[MctsNode] = None,
        llm_heuristic: Any = None,
    ):
        self._state = state
        self._config = config
        self._expansion_policy = config.expansion_policy
        self._filter_policy = config.filter_policy
        self.tree = owner
        self.is_expanded: bool = False
        self.is_expandable: bool = not self.state.is_terminal
        self._parent = parent

        if owner is None:
            self.created_at_iteration: Optional[int] = None
        else:
            self.created_at_iteration = self.tree.profiling["iterations"]

        self._children_values: List[float] = []
        self._children_priors: List[float] = []
        self._children_visitations: List[int] = []
        self._children_actions: List[RetroReaction] = []
        self._children_heuristic_scores: List[float] = []  # for LLM-guided expansion
        self._children: List[Optional[MctsNode]] = []

        self._synthesis_dynamic_context: SynthesisPlanUpdate = None
        self._has_solution = None

        self.blacklist = set(mol.inchi_key for mol in state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

        if self._algo_config["mcts_grouping"]:
            self._degeneracy_check = self._algo_config["mcts_grouping"].lower()
        else:
            self._degeneracy_check = "none"
        self._logger = logger()

        # Steer
        # if llm_heuristic:
        #     self.llm_heuristic = llm_heuristic
        # elif parent:
        #     self.llm_heuristic = parent.llm_heuristic
        # self.heuristic_weight = config.search.steer.llm_weight
        # self.heuristic_prob = config.search.steer.llm_prob

        # LLM Guidance Expansion config and policy without steer integration (would be doubled)
        if (
            hasattr(config.search, "llm_guidance")
            and config.search.llm_guidance.enabled
        ):
            scorer_config = config.search.llm_guidance.strategic_scorer
            self.heuristic_weight = scorer_config.get("weight", 1.0)
            self.heuristic_prob = scorer_config.get("probability", 1.0)
        else:
            self.heuristic_weight = 1.0  # defaults
            self.heuristic_prob = 1.0

        # print(f" LLM Action Scoring: prob={self.heuristic_prob}, weight={self.heuristic_weight}")

    def __getitem__(self, node: "MctsNode") -> StrDict:
        idx = self._children.index(node)
        return {
            "action": self._children_actions[idx],
            "value": self._children_values[idx],
            "prior": self._children_priors[idx],
            # "heuristic_score": self._children_heuristic_scores[idx],
            "visitations": self._children_visitations[idx],
        }

    @classmethod
    def create_root(
        cls, smiles: str, tree: MctsSearchTree, config: Configuration
    ) -> "MctsNode":
        """
        Create a root node for a tree using a SMILES.

        :param smiles: the SMILES representation of the root state
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :return: the created node
        """
        mol = TreeMolecule(parent=None, transform=0, smiles=smiles)
        state = MctsState(mols=[mol], config=config)
        return cls(state=state, owner=tree, config=config)

    @classmethod
    def from_dict(
        cls,
        dict_: StrDict,
        tree: MctsSearchTree,
        config: Configuration,
        molecules: MoleculeDeserializer,
        parent: Optional["MctsNode"] = None,
        llm_heuristic: Any = None,
    ) -> "MctsNode":
        """
        Create a new node from a dictionary, i.e. deserialization.

        :param dict_: the serialized node
        :param tree: the search tree
        :param config: settings of the tree search algorithm
        :param molecules: the deserialized molecules
        :param parent: the parent node
        :return: a deserialized node
        """
        # pylint: disable=protected-access
        state = MctsState.from_dict(dict_["state"], config, molecules)
        node = cls(state=state, owner=tree, config=config, parent=parent)
        node.is_expanded = dict_["is_expanded"]
        node.is_expandable = dict_["is_expandable"]
        node._children_values = dict_["children_values"]
        node._children_priors = dict_["children_priors"]
        node._children_visitations = dict_["children_visitations"]
        node._children_heuristic_scores = dict_.get("children_heuristic_scores", [])
        node._children_actions = [
            deserialize_action(action_dict, molecules)
            for action_dict in dict_["children_actions"]
        ]
        node._children = [
            cls.from_dict(
                child, tree, config, molecules, parent=node, llm_heuristic=llm_heuristic
            )
            if child
            else None
            for child in dict_["children"]
        ]
        if dict_.get("synthesis_dynamic_context", None):
            node._synthesis_dynamic_context = SynthesisPlanUpdate.from_dict(
                dict_["synthesis_dynamic_context"]
            )
        return node

    @property
    def config(self):
        return self.tree.config

    @property
    def children(self) -> List["MctsNode"]:
        """
        Returns all of the instantiated children.

        :return: the children
        """
        return [child for child in self._children if child]

    @property
    def is_solved(self) -> bool:
        """Return if the state is solved."""
        return self.state.is_solved

    @property
    def parent(self) -> Optional["MctsNode"]:
        """Return the parent of the node."""
        return self._parent

    @property
    def state(self) -> MctsState:
        """Return the underlying state of the node."""
        return self._state

    @property
    def state_smiles(self) -> str:
        """Return the SMILES representation of the state."""
        return [each.smiles for each in self.state.mols]

    @property
    def context_for_llm(self) -> LLMGuidedNodeContext:
        """Return the context of the node, if available."""
        # strategy = self.tree._synthesis_plan
        # target_molecule = self.tree.root.state.mols[0]
        # reaction_path = self.get_smiles_tree()
        # current_depth = self.calculate_path_depth()
        # return {
        #     'target_molecule': target_molecule.smiles,
        #     'synthesis_plan': strategy,
        #     'reaction_path': reaction_path,
        #     'depth': current_depth,
        #     'current_molecules': [mol.smiles for mol in self.state.expandable_mols],
        #     'tree': self.tree
        # }
        synthesis_context = self.tree._synthesis_context
        return LLMGuidedNodeContext(
            reaction_path=self.get_smiles_tree(),
            depth=self.calculate_path_depth(),
            synthesis_context=synthesis_context,
            tree=self.tree,
        )

    @property
    def synthesis_context(self):
        return self.tree._synthesis_context

    @property
    def synthesis_dynamic_context(self):
        return self._synthesis_dynamic_context

    @property
    def user_constraint(self):
        return self.context_for_llm.synthesis_context.user_constraint

    @property
    def _algo_config(self) -> StrDict:
        """Just a convinient, shorter name of this."""
        return self._config.search.algorithm_config

    @property
    def target_molecule(self) -> TreeMolecule:
        return self.tree.root.state.mols[0]

    @property
    def reaction_str(self):
        if self.get_smiles_tree():
            return self.get_smiles_tree()[-1]
        else:
            return None

    @property
    def has_solution(self):
        if not self.children:
            return self.is_solved
        return any([child.has_solution for child in self.children])

    def get_descendants(self) -> List["MctsNode"]:
        descendants = [self]
        descendants.extend(self.children)
        for child in self.children:
            descendants.extend(child.get_descendants())
        return descendants

    def get_leaf_nodes(self) -> List["MctsNode"]:
        leaf_nodes = []
        for node in self.get_descendants():
            if not node.children:
                leaf_nodes.append(node)
        return leaf_nodes

    def children_view(self) -> StrDict:
        """
        Creates a view of the children attributes. Each of the
        list returned is a new list, although the actual children
        are not copied.

        The return dictionary will have keys "actions", "values",
        "priors", "visitations" and "objects".

        :return: the view
        """
        return {
            "actions": list(self._children_actions),
            "values": list(self._children_values),
            "priors": list(self._children_priors),
            "visitations": list(self._children_visitations),
            "heuristic_scores": list(self._children_heuristic_scores),
            "objects": list(self._children),
        }

    # def _pass_context_to_policy(self):

    async def _expansion_policy_call(
        self,
        # expandable_mols: List[TreeMolecule],
        cache_molecules: List[TreeMolecule],
    ) -> Tuple[List[RetroReaction], List[float]]:
        """Wrapper for the self._expansion_policy call to handle extra output from LLM-guided expansion"""
        # for expansion_strat in self._expansion_policy._items:
        if getattr(self._expansion_policy, "ignore_stock", False):
            expandable_mols = self.state.mols
        else:
            expandable_mols = self.state.expandable_mols

        if hasattr(self._expansion_policy, "get_actions_full_response"):
            (
                actions,
                priors,
                synthesis_plan,
            ) = await self._expansion_policy.get_actions_full_response(
                expandable_mols,
                cache_molecules,
            )
            synthesis_plan = synthesis_plan[0] if synthesis_plan else None
            if synthesis_plan:
                self._synthesis_dynamic_context = synthesis_plan
            self._logger.debug(
                f"Synthesis context updated for node: {self.state_smiles}"
            )
        else:
            actions, priors = self._expansion_policy(
                expandable_mols,
                cache_molecules,
            )
            self._logger.info(
                f"Expansion without updating synthesis context: {self.state_smiles}"
            )
        return actions, priors

    async def expand_async(self) -> None:
        """
        Standard MCTS expansion using configured expansion policy.
        """
        if self.is_expanded:
            msg = f"Oh no! This node is already expanded. id={id(self)}"
            self._logger.error(msg)
            raise NodeUnexpectedBehaviourException(msg)

        if not self._expansion_policy.ignore_stock and (
            self.is_expanded or not self.is_expandable
        ):
            print(
                f"Cannot expand: expanded={self.is_expanded}, expandable={self.is_expandable}"
            )
            return

        self.is_expanded = True

        cache_molecules = []
        if self.parent:
            for child in self.parent.children:
                cache_molecules.extend(child.state.mols)

        ### synthelite
        try:
            self._pass_context_to_policy()

        except Exception as e:
            self._logger.error(f"Failed to build context: {e}")
            raise e
        ###

        # TODO: Log the raw response of next step generator somehow
        actions, priors = await self._expansion_policy_call(
            # self.state.expandable_mols,
            cache_molecules
        )

        self._fill_children_lists(actions, priors)

        if len(actions) == 0:
            # print("No actions generated, node is not expandable")
            self.is_expandable = False
            self.is_expanded = False

        if self.tree:
            self.tree.profiling["expansion_calls"] += 1

        if not self._algo_config["immediate_instantiation"]:
            return
        # Instantiate all children actions created by the marked policy,
        # a new list of actions will be iterated over, because it can grow due
        # to instantiation
        nactions = len(actions)
        for child_idx, action in enumerate(self._children_actions[:nactions]):
            policy_name = action.metadata.get("policy_name")
            if (
                policy_name
                and policy_name in self._algo_config["immediate_instantiation"]
            ):
                self._instantiate_child(child_idx)

        self._deduplicate_children()

    def expand(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.expand_async(*args, **kwargs))

    def _deduplicate_children(self):
        seen_states = set()
        for i in range(len(self._children)):
            if self._children[i] is None:
                continue
            child = self._children[i]
            if child.state in seen_states:
                self._children[i] = None
                self._children_values[i] = -1e6  # this child is disabled
                self._children_priors[i] = 0
                self._children_visitations[i] = 0
                # self._children_actions[i] = None
                if self._children_heuristic_scores:
                    self._children_heuristic_scores[i] = 0
            else:
                seen_states.add(child.state)
        # Remove None values from the lists

    def _pass_context_to_policy(self):
        """Pass context to LLM-guided expansion policy"""
        try:
            # if hasattr(self._expansion_policy, 'selection') and hasattr(self._expansion_policy, '_items'):
            #     for policy_name in self._expansion_policy.selection:
            for policy_name, policy in self._expansion_policy._items.items():
                # policy = self._expansion_policy._items.get(policy_name)
                # if hasattr(policy, 'key') and policy.key == "llm_guided":
                if hasattr(policy, "set_node_context"):
                    # context = self.context_for_llm
                    policy.set_node_context(self)
                    self._logger.debug(f" Context passed to LLM policy")
                # else:
                #     print(f"    LLM policy has no set_node_context method")
                # break
        except Exception as e:
            self._logger.error(f"Error passing context to LLM policy: {e}")
            raise e

    def calculate_path_depth(self) -> int:
        """Calculate the depth of this node in the search tree"""
        depth = 0
        current = self
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    # def is_terminal(self) -> bool:
    #     """
    #     Node is terminal if its unexpandable, or the internal state is terminal (solved).

    #     :return: the terminal attribute of the node
    #     """
    #     return not self.is_expandable or self.state.is_terminal

    # def path_to(self) -> Tuple[List[RetroReaction], List[MctsNode]]:
    #     """
    #     Return the path to this node, which is a list of actions and a list of node.

    #     :return: the actions and nodes
    #     """
    #     return route_to_node(self)

    def promising_child(self) -> Optional["MctsNode"]:
        """
        Return the child with the currently highest Q+U.

        The selected child will be instantiated if it has not been already.

        If no actions could be found that were applicable, the method will
        return None.

        :return: the child
        """
        child = None
        while child is None:
            try:
                child = self._score_and_select()
            # _score_and_select raises exception if no children can be selected
            except ValueError:
                child = None
                break

        if not child:
            self._logger.debug(
                "Returning None from promising_child() because there were no applicable action"
            )
            self.is_expanded = False
            self.is_expandable = False

        return child

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the node object to a dictionary.

        :param molecule_store: the serialized molecules
        :return: the serialized node
        """
        return {
            "state": self.state.serialize(molecule_store),
            "children_values": self._serialize_stats_list("_children_values"),
            "children_priors": self._serialize_stats_list("_children_priors"),
            "children_heuristic_scores": self._serialize_stats_list(
                "_children_heuristic_scores"
            ),
            "children_visitations": self._children_visitations,
            "children_actions": [
                serialize_action(action, molecule_store)
                for action in self._children_actions
            ],
            "children": [
                child.serialize(molecule_store) if child else None
                for child in self._children
            ],
            "is_expanded": self.is_expanded,
            "is_expandable": self.is_expandable,
            "synthesis_dynamic_context": self._synthesis_dynamic_context.to_dict()
            if self._synthesis_dynamic_context
            else None,
        }

    def to_reaction_tree(self) -> ReactionTree:
        """
        Return reaction tree from the path of actions and nodes leading to this node.

        :return: the constructed tree
        """
        return ReactionTreeFromSuperNode(self).tree

    def _check_child_reaction(self, reaction: RetroReaction) -> bool:
        if not reaction.reactants:
            self._logger.debug(f"{reaction} did not produce any reactants")
            return False

        # fmt: off
        reactants0 = reaction.reactants[0]
        if len(reaction.reactants) == 1 and len(reactants0) == 1 and reaction.mol == reactants0[0]:
            return False
        # fmt: on

        return True

    def _children_q(self) -> np.ndarray:
        return np.array(self._children_values) / np.array(self._children_visitations)

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._algo_config["C"] * np.sqrt(2 * total_visits / child_visits)

    def _create_children_nodes(
        self, states: List[MctsState], child_idx: int
    ) -> List["MctsNode"]:
        new_nodes = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            if self._generated_degeneracy(state, first_child_idx):
                # Only need to disable first new child,
                # if the action generated more states, we will just not generate
                # a child for that state
                if state_index == 0:
                    self._disable_child(child_idx)
                continue

            # If there's more than one outcome, the lists need be expanded
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            if self._filter_child_reaction(self._children_actions[child_idx]):
                self._disable_child(child_idx)
            else:
                new_node = self.__class__(
                    state=state, owner=self.tree, config=self._config, parent=self
                )
                self._children[child_idx] = new_node
                new_nodes.append(new_node)
        return new_nodes

    def _disable_child(self, child_idx: int) -> None:
        self._children_values[child_idx] = -1e6

    def _delete_child(self, child_idx: int):
        if not self._children:
            self._logger.warning("No children to delete")
        self._children[child_idx] = None

    def _purge_children(self, keep_children: List[MctsNode] = []):
        for child_id, child_node in enumerate(self._children):
            if child_node not in keep_children:
                self._delete_child(child_id)
                self._disable_child(child_id)
                self._logger.debug(f"Purged child {child_id} from node {id(self)}")

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        new_action = self._children_actions[old_index].copy(index=action_index)
        self._children_actions.append(new_action)
        self._children_priors.append(self._children_priors[old_index])
        self._children_values.append(self._children_values[old_index])
        if self._children_heuristic_scores:
            self._children_heuristic_scores.append(
                self._children_heuristic_scores[old_index]
            )
        self._children_visitations.append(self._children_visitations[old_index])
        self._children.append(None)
        return len(self._children) - 1

    def _fill_children_lists(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> None:
        self._children_actions = actions
        self._children_priors = priors
        nactions = len(actions)
        self._children_visitations = [1] * nactions
        self._children_heuristic_scores = [
            0.0
        ] * nactions  # Initialize heuristic scores
        self._children = [None] * nactions
        if self._algo_config["use_prior"]:
            self._children_values = list(self._children_priors)
        else:
            self._children_values = [self._algo_config["default_prior"]] * nactions

    def _filter_child_reaction(self, reaction: RetroReaction) -> bool:
        # if self._filter_actions_incompatibilities(reaction):
        #     return True

        if self._regenerated_blacklisted(reaction):
            self._logger.debug(
                f"Reaction {reaction.reaction_smiles()} "
                f"was rejected because it re-generated molecule not in stock"
            )
            return True

        if not self._filter_policy.selection:
            return False
        try:
            self._filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _generated_degeneracy(self, new_state: MctsState, child_idx: int) -> bool:
        """
        Check if a new MCTS state is equal to another MCTS state of a children node.

        The check can be "partial" in which the equality is based only on the expandable molecules,
        or "full" in which the equality is based on all molecules in the state.

        The comparison will not be made on unexpanded children nodes
        or terminal children nodes.

        The metadata of the degenerate action will be added to the metadata
        of the previously created equal state.
        """

        def equal_states(query_state):
            if self._degeneracy_check == "partial":
                return query_state.expandables_hash == new_state.expandables_hash
            return query_state == new_state

        if self._degeneracy_check not in ["partial", "full"]:
            return False
        previous_action = None
        for child, action in zip(self._children, self._children_actions):
            if (
                child is not None
                and not child.is_terminal()  # method to property changed
                and equal_states(child.state)
            ):
                previous_action = action
                break

        if previous_action is None:
            return False

        # No need to copy the metadata because it will be the same
        if previous_action is self._children_actions[child_idx]:
            return True

        metadata_copy = dict(self._children_actions[child_idx].metadata)
        if "additional_actions" not in previous_action.metadata:
            previous_action.metadata["additional_actions"] = []
        previous_action.metadata["additional_actions"].append(metadata_copy)
        return True

    def _instantiate_child(self, child_idx: int) -> List["MctsNode"]:
        """
        Instantiate the children node.

        The algorithm is:
        * Apply the reaction associated with the child
        * If the application of the action failed, set value to -1e6 and return None
        * Create a new state array, one new state for each of the reaction outcomes
        * Create new child nodes
            - If a filter policy is available and the reaction outcome is unlikely
              set value of child to -1e6
         * Return all new nodes
        """
        if self._children[child_idx] is not None:
            raise NodeUnexpectedBehaviourException("Node already instantiated")

        reaction = self._children_actions[child_idx]
        if reaction.unqueried:
            if self.tree:
                self.tree.profiling["reactants_generations"] += 1
            _ = reaction.reactants

        if not self._check_child_reaction(reaction):
            self._disable_child(child_idx)
            return []

        keep_mols = [mol for mol in self.state.mols if mol is not reaction.mol]
        new_states = [
            MctsState(keep_mols + list(reactants), self._config)
            for reactants in reaction.reactants
        ]

        # Create children nodes First
        new_nodes = self._create_children_nodes(new_states, child_idx)

        # Then add LLM context to the molecules in the new nodes
        reaction = self._children_actions[child_idx]
        next_step_query = reaction.metadata.get("next_step_query")
        strategic_alignment = reaction.metadata.get("strategic_alignment")

        for new_node in new_nodes:
            for mol in new_node.state.expandable_mols:
                mol.llm_next_step_query = next_step_query
                mol.strategic_alignment = strategic_alignment

        return new_nodes

    def _regenerated_blacklisted(self, reaction: RetroReaction) -> bool:
        if not self._algo_config["prune_cycles_in_search"]:
            return False
        for reactants in reaction.reactants:
            for mol in reactants:
                if mol.inchi_key in self.blacklist:
                    return True
        return False

    # def _score_and_select(self) -> Optional["MctsNode"]:
    #     if not max(self._children_values) > 0:
    #         raise ValueError("Has no selectable children")
    #     scores = self._children_q() + self._children_u()
    #     heuristic_values = asyncio.run(self.get_heuristic_values_async())
    #     heuristic_values = np.array(heuristic_values, dtype=float)
    #     scores = scores + self.heuristic_weight * heuristic_values

    #     indices = np.where(scores == scores.max())[0]
    #     index = np.random.choice(indices)

    #     # Show which action was selected
    #     self._logger.debug(f" Selected Action {index} with final score {scores[index]:.3f}")

    #     return self._select_child(index)

    def _greedy_score_and_select(self) -> Optional["MctsNode"]:
        """Select the child with the highest LLM score without exploration."""
        # scores = asyncio.run(self.get_heuristic_values_async())
        scores = self.get_heuristic_values()

        index = np.argmax(scores)

        # Show which action was selected
        self._logger.info(
            f" Selected Action {index} with LLM score {scores[index]:.3f}"
        )

        return self._select_child(index)

    async def get_heuristic_values_async(self):
        return self.get_heuristic_values()

    def get_heuristic_values(self):
        """score actions using LLM-guided heuristic
        TODO: Might be a good point to implement the context cache for scoring.
        """

        # better than node scoring as node scoring requires instantiation of expanded children

        # cache_key = f"heuristic_cache_{len(self._children_actions)}"

        # if hasattr(self, cache_key):
        #     print(f" Using cached heuristic values for {len(self._children_actions)} actions")
        #     return getattr(self, cache_key)
        if self._children_heuristic_scores:
            return np.array(self._children_heuristic_scores, dtype=float)

        self._logger.debug("get_heuristic_values() is called")
        self._logger.debug(f"Node depth: {self.calculate_path_depth()}")
        self._logger.debug(f"Node has parent: {self._parent is not None}")
        self._logger.debug(f"Actions count: {len(self._children_actions)}")

        current_path = self.get_smiles_tree()
        self._logger.debug(f"Current reaction path length: {len(current_path)}")

        heuristic_score_fn_name = self._config.search.llm_guidance.strategic_scorer[
            "score_function"
        ]
        heuristic_score_fn = HEURISTIC_SCORER[heuristic_score_fn_name]

        return heuristic_score_fn([self])

    def get_smiles_tree(self, mapped=False, retro=True):
        molist = []
        current = self
        while current is not None:
            if mapped:
                smis = ".".join([m.mapped_smiles for m in current.state.mols])
            else:
                smis = ".".join([m.smiles for m in current.state.mols])
            molist.append(smis)
            current = current.parent
        molist = list(reversed(molist))

        if retro:
            rxns = [f"{molist[i]}>>{molist[i+1]}" for i in range(len(molist) - 1)]
        else:
            rxns = [f"{molist[i+1]}>>{molist[i]}" for i in range(len(molist) - 1)]

        # Clean rxns: if prod in reactants rm
        crxns = []
        for r in rxns:
            reacts = r.split(">>")[0].split(".")
            prodts = r.split(">>")[1].split(".")
            if not any([p in reacts for p in prodts]):
                crxns.append(r)
            else:
                nrts = [r for r in reacts if r not in prodts]
                npts = [p for p in prodts if p not in reacts]
                crxns.append(f"{'.'.join(nrts)}>>{'.'.join(npts)}")

        return crxns

    def _filter_actions_incompatibilities(self, reaction: RetroReaction) -> bool:
        if reaction.metadata.get("policy_name", "") == "onmt":
            return False
        reactants = ".".join(sorted(r.smiles for r in reaction.reactants[0]))
        tmplt = reaction.smarts
        pot_prods = check_template(tmplt, reactants)
        if len(pot_prods) > 1:
            return True
        elif len(pot_prods) == 1:
            # If product from template application is not the same as expected product.
            if canon_mol(list(pot_prods)[0]) != canon_mol(reaction.mol.smiles):
                return True
        return False

    def _filter_unproductive_cycles(self, scores) -> List:
        """Look up proposed reactions and compare with current reaction, to avoid unproductive cycles."""

        def is_unproductive_cycle(f_temp, this_temp):
            return f_temp == this_temp

        # Make sure self.parent.state.mols is always only one
        if self.parent is not None:
            # assert len(self.parent.state.mols) == 1

            parent_mols = ".".join([m.mapped_smiles for m in self.parent.state.mols])
            reacts = ".".join([m.mapped_smiles for m in self.state.mols])
            rxn = ChemicalReaction(f"{reacts}>>{parent_mols}")
            try:
                f_temp = rxn.generate_reaction_template(radius=1)[
                    1
                ].smarts  # 0 for canonical, 1 for retro
                # todo: calc reaction type of this template
            except ReactionException:
                return scores

            for i, a in enumerate(self._children_actions):
                this_temp = a.metadata["template"]
                # todo: calc reaction type of this template

                if is_unproductive_cycle(f_temp, this_temp):
                    print("wassup homie")
                    scores[i] = 0.0

        return scores

    def _select_child(self, child_idx: int) -> Optional["MctsNode"]:
        """
        Selecting a child node implies instantiating the children nodes.

        If the child has already been instantiated, return immediately
        Otherwise, select a random node of the feasible ones to return
        """
        if self._children[child_idx]:
            return self._children[child_idx]

        new_nodes = self._instantiate_child(child_idx)
        if new_nodes:
            return random.choice(new_nodes)
        return None

    def _serialize_stats_list(self, name: str) -> List[float]:
        return [float(value) for value in getattr(self, name)]

    def reset_expansion(self):
        super().reset_expansion()
        self._children_heuristic_scores = []

    def get_subtree_leaf_nodes(self):
        if not self.children:
            return []
        nodes = []
        for child in self.children:
            if not child.children:
                nodes.append(child)
            else:
                nodes.extend(child.get_subtree_leaf_nodes())
        return nodes

    def _update_children_values_from_heuristic(self):
        heuristic_scores = []
        branching_weight = 5  # Fixed value. Has no meaningful effect in greedy search.
        for heuristic_score, children_node in zip(
            self._children_heuristic_scores, self._children
        ):
            if children_node is not None:
                heuristic_scores.append(heuristic_score)
            else:
                heuristic_scores.append(0.0)

        children_values = normalize_heuristic_scores(heuristic_scores)

        for i, (children_node, value) in enumerate(
            zip(self._children, children_values)
        ):
            if children_node is not None:
                branching_factor = len(
                    children_node.get_subtree_leaf_nodes()
                )  # higher branching factor means the LLM prefers to explore that route.
                children_values[i] = value + branching_weight * branching_factor

        self._children_values = children_values

    def _zero_children_values_initialize(self):
        values = []
        for children_node in self._children:
            if children_node is not None:
                values.append(0)
            else:
                values.append(-1e6)
        self._children_values = values

    def initialize_children_values(self, from_heuristic: bool = True):
        if from_heuristic:
            self._update_children_values_from_heuristic()
        else:
            self._zero_children_values_initialize()


class ParetoMctsNode(MctsNode):
    """
    A node in a multi-objective tree search.

    This implements the algorithm from:
        Chen W., Liu L. Pareto Monte Carlo Tree Search for Multi-Objective Informative Planning
        Robotics: Science and Systems 2019, 2012 arXiv:2111.01825


    The main difference compared to the standard MCTS algorithm is:
        - Children stats: the values, cumulative reward and priors are nested
                     list, with one value per objective
        - Selection: children on Pareto front are computed, and
                     a child from this set is taken randomly

    This implementation disregards the prior of the children when it has been
    visited once.

    It is assumed that all objectives are to be maximised.
    """

    def __init__(
        self,
        state: MctsState,
        owner: MctsSearchTree,
        config: Configuration,
        parent: Optional[ParetoMctsNode] = None,
    ):
        super().__init__(state, owner, config, parent)
        self._num_objectives = len(self._algo_config["search_rewards"])
        self._prior_weight = 1
        self._direction = "max"  # current implementation assumes maximisation
        self._children_rewards_cummulative: List[List[float]]
        self._children_values: List[List[float]]  # type: ignore
        self._children_priors: List[List[float]]  # type: ignore

    def backpropagate(self, child: "MctsNode", value_estimate: List[float]) -> None:  # type: ignore
        """
        Update the number of visitations of a particular child and its value.

        :param child: the child node
        :param value_estimate: the value to add to the child value
        """
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        # here we only update the cummulative rewards,
        #  _children_values are updated at selection time
        new_value_estimate = [
            cum_reward + new_reward
            for cum_reward, new_reward in zip(
                self._children_rewards_cummulative[idx], value_estimate
            )
        ]
        self._children_rewards_cummulative[idx] = new_value_estimate

    def children_view(self) -> StrDict:
        """
        Creates a view of the children attributes. Each of the
        list returned is a new list, although the actual children
        are not copied.

        The return dictionary will have keys "actions", "values",
        "priors", "visitations", "rewards_cum", and "objects".

        :return: the view
        """
        dict_ = super().children_view()
        dict_["rewards_cum"] = list(self._children_rewards_cummulative)
        return dict_

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the node object to a dictionary.

        :param molecule_store: the serialized molecules
        :return: the serialized node
        """
        dict_ = super().serialize(molecule_store)
        dict_["children_cumulative_reward"] = self._serialize_stats_list(
            "_children_rewards_cummulative"
        )
        return dict_

    def _disable_child(self, child_idx: int) -> None:
        self._children_rewards_cummulative[child_idx] = [-1e6] * self._num_objectives

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        ret = super()._expand_children_lists(old_index, action_index)
        # These lists are nested lists, so need to make sure that the inner lists are also new
        self._children_values[-1] = list(self._children_values[-1])
        self._children_priors[-1] = list(self._children_priors[-1])
        self._children_rewards_cummulative.append(
            list(self._children_rewards_cummulative[old_index])
        )
        return ret

    def _fill_children_lists(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> None:
        self._children_actions = actions
        nactions = len(actions)
        # shape: num_actions x 1
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        # shape: num_actions x num_objectives
        self._children_rewards_cummulative = [[0.0] * self._num_objectives] * nactions
        if self._algo_config["use_prior"]:
            # shape: num_actions x num_objectives
            # for children i, 3 objectives -> [prior i, prior i, prior i]
            self._children_priors = [[prior] * self._num_objectives for prior in priors]

        else:
            self._children_priors = [
                [self._algo_config["default_prior"]] * self._num_objectives
            ] * nactions

        # at initialisation, values = prior as cummulative rewards are zero
        self._children_values = [
            [prior * self._prior_weight for prior in priors]
            for priors in self._children_priors
        ]

    def _children_q(self, children_values_arr):
        children_visitations_expanded = np.repeat(
            np.array(self._children_visitations).reshape(-1, 1),
            axis=1,
            repeats=self._num_objectives,
        )
        return children_values_arr / children_visitations_expanded

    def _compute_children_scores(self) -> np.ndarray:
        """Compute the modified ucb scores: alpha * prior + average reward + exploration."""
        # update prior to zero once the node has been visited
        children_priors_arr = self._prior_schedule_oneoff()
        # compute prior_weight * prior + cummulative rewards
        children_values_arr = self._prior_weight * children_priors_arr + np.array(
            self._children_rewards_cummulative
        )
        expanded_u = np.repeat(
            self._children_u().reshape(-1, 1), axis=1, repeats=self._num_objectives
        )
        # _children_scores shape: num_childrens x num_objectives
        children_scores = self._children_q(children_values_arr) + expanded_u
        if children_scores.shape[1] != self._num_objectives:
            raise ValueError(
                f"expected second dimension to have {self._num_objectives},"
                f"currently has {children_scores.shape[1]}"
            )
        self._children_values = children_values_arr.tolist()
        self._children_priors = children_priors_arr.tolist()

        return children_scores

    def _prior_schedule_oneoff(self) -> np.ndarray:
        # shape: num_children x 1
        visted_mask = (np.array(self._children_visitations) > 1).reshape(-1, 1)
        # shape: num_children x num_objectives
        visted_mask = np.repeat(visted_mask, axis=1, repeats=self._num_objectives)
        # set the prior weights for visited children to be zero
        children_priors_arr = np.array(self._children_priors)
        children_priors_arr[visted_mask] = 0
        return children_priors_arr

    def _score_and_select(self) -> Optional["MctsNode"]:
        if not max(max(value_list) for value_list in self._children_values) > 0:
            raise ValueError("Has no selectable children")
        children_scores = self._compute_children_scores()
        pareto_idxs = self._update_pareto_front(children_scores)
        index = np.random.choice(pareto_idxs)
        return self._select_child(index)

    def _serialize_stats_list(self, name: str):
        return [
            [float(value) for value in value_list] for value_list in getattr(self, name)
        ]

    def _update_pareto_front(self, children_scores: np.ndarray) -> np.ndarray:
        """
        Update the pareto front of a node, this step normally happens
        after its values for the best child have been updated.

        :param children_scores: Children scores
        :returns: Pareto front children indexes
        """
        direction_arr = np.repeat(self._direction, self._num_objectives)
        mask = paretoset(children_scores, sense=direction_arr, distinct=False)
        return np.arange(len(self._children))[mask]
