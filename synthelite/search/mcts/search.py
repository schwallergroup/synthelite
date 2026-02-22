""" Module containing a class that holds the tree search
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from typing import Any

from synthelite.agent_search.schema import (
    PlanningAttempt,
    SynthesisContext,
    SynthesisPlanUpdate,
)
from synthelite.chem import MoleculeDeserializer, MoleculeSerializer
from synthelite.chem.reaction import action_to_str
from synthelite.context.policy.expansion_strategies import (
    DynamicLLMGuidedExpansionStrategy,
    MultiExpansionPolicyLLMFallback,
)
from synthelite.search.mcts.heuristic import (
    HEURISTIC_SCORER,
    get_route_heuristic_score_fn,
)
from synthelite.search.mcts.node import MctsNode, ParetoMctsNode, LLMGuidedMctsNode
from synthelite.utils.clogging import logger, init_weave
import os
import sys
import weave

from networkx.drawing.nx_agraph import graphviz_layout


if TYPE_CHECKING:
    from synthelite.context.config import Configuration
    from synthelite.utils.type_utils import List, Optional, Sequence, Union


_MODE2NODECLASS = {
    "llm-guided": LLMGuidedMctsNode,
    "single-objective": MctsNode,
    "weighted-sum": MctsNode,
    "multi-objective": ParetoMctsNode,
}

from synthelite.utils.reaction_tree import deduplicate_routes

from dotenv import load_dotenv
import os

load_dotenv()
wandb_project = os.getenv("WANDB_PROJECT", "synthelite")
init_weave(wandb_project)


class MctsSearchTree:
    """
    Encapsulation of the search tree.

    :ivar root: the root node
    :ivar config: the configuration of the search tree

    :param config: settings of the tree search algorithm
    :param root_smiles: the root will be set to a node representing this molecule, defaults to None
    """

    def __init__(
        self,
        config: Configuration,
        root_smiles: Optional[str] = None,
        steer_query: Optional[str] = None,
        previous_attempts: List[PlanningAttempt] = [],
        strategy_json: Optional[str] = None,
    ) -> None:
        """steer_query will be ignored if strategy_json is provided"""
        self._logger = logger()
        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
            "iterations": 0,
        }
        self.config = config
        self.mode = self._check_mode()
        self._logger.debug(f"MCTS mode: {self.mode}")
        # Initialize LLM guidance only if enabled in config

        if root_smiles:
            self.root: Optional[MctsNode] = _MODE2NODECLASS[self.mode].create_root(
                smiles=root_smiles, tree=self, config=config
            )
        else:
            self.root = None

        self._synthesis_context: Optional[SynthesisContext] = None
        self._initialize_synthesis_context(
            self.root_smiles, steer_query, previous_attempts, strategy_json
        )

        self._graph: Optional[nx.DiGraph] = None

        # For backward compatibility
        if "search_reward" in config.search.algorithm_config:
            config.search.algorithm_config["search_rewards"] = [
                config.search.algorithm_config.pop("search_reward")
            ]
        config_rewards = config.search.algorithm_config["search_rewards"]
        self._logger.setLevel(logging.WARNING)  # Supress logging from `make_subset`
        self._logger.debug(f"Selecting reward scorers: {config_rewards}")
        self.reward_scorer = self.config.scorers.make_subset(config_rewards)
        self.reward_scorer_name = config_rewards[0]
        self._logger.setLevel(logging.DEBUG)
        if self.mode == "single-objective":
            if len(config_rewards) > 1:
                self._logger.warning(
                    "Single-objective search but multiple rewards specified - "
                    "Use multi-objective mode for multiple rewards"
                )

    @property
    def root_smiles(self):
        return self.root.state.mols[0].smiles if self.root else None

    def _initialize_synthesis_context(
        self,
        root_smiles: str,
        steer_query: str,
        previous_attempts: List[PlanningAttempt],
        strategy_json,
    ) -> SynthesisContext:
        if isinstance(
            self.config.expansion_policy._items.get("llm_guided"),
            DynamicLLMGuidedExpansionStrategy,
        ):
            is_strategy_dynamic = True
        elif isinstance(
            self.config.expansion_policy._items.get("llm_guided"),
            MultiExpansionPolicyLLMFallback,
        ):
            is_strategy_dynamic = True
        else:
            is_strategy_dynamic = False

        # Initialize LLM guidance only if enabled in config
        if (
            hasattr(self.config.search, "llm_guidance")
            and self.config.search.llm_guidance.enabled
        ):
            if strategy_json:
                self._get_synthesis_plan_from_file(strategy_json)
            if root_smiles and steer_query:
                if is_strategy_dynamic:
                    self._create_minimal_synthesis_context(
                        root_smiles, steer_query, previous_attempts
                    )
                else:
                    self._create_synthesis_plan_directly(
                        root_smiles, steer_query, previous_attempts
                    )

    def _create_minimal_synthesis_context(
        self,
        target_smiles: str,
        user_constraint: str,
        previous_attempts: List[PlanningAttempt],
    ):
        """Create a minimal synthesis context for dynamic strategy"""
        self._synthesis_context = SynthesisContext(
            target_smiles=target_smiles,
            user_constraint=user_constraint,
            previous_attempts=previous_attempts,
            # expandable_molecules=[target_smiles],  # Start with the target molecule
        )
        self._logger.debug(f"Synthesis context: {self._synthesis_context}")

    def _get_synthesis_plan_from_file(self, filename: str) -> Optional[dict]:
        with open(filename, "r") as fileobj:
            context = json.load(fileobj)
        self._synthesis_context = context

    def _create_synthesis_plan_directly(self, target_smiles: str, user_constraint: str):
        """Create synthesis plan directly using LLM modules"""
        try:
            from synthelite.agent_search.strategy_planner import (
                SynthesisStrategyPlanner,
            )

            self._logger.info(
                f"Creating synthesis plan for LLM-guided search. config: {user_constraint}"
            )
            self._logger.info("Creating synthesis plan for LLM-guided search.")

            planner_kwargs = {"temperature": 0.1, "verbose": False}
            llm_guidance_cfg = getattr(self.config.search, "llm_guidance", None)
            if llm_guidance_cfg and getattr(llm_guidance_cfg, "strategy_planner", None):
                strategy_cfg = llm_guidance_cfg.strategy_planner
                if isinstance(strategy_cfg, dict):
                    planner_kwargs.update(strategy_cfg)
                else:
                    # dataclass-like object
                    planner_kwargs.update(
                        {
                            k: v
                            for k, v in strategy_cfg.__dict__.items()
                            if not k.startswith("_")
                        }
                    )
            model = planner_kwargs.pop("model", "claude-3-5-sonnet-20241022")
            planner = SynthesisStrategyPlanner(model=model, **planner_kwargs)
            strategy = planner.create_strategy(target_smiles, user_constraint)

            self._synthesis_context = strategy

            self._logger.info("Synthesis plan created for search tree")
            self._logger.debug(
                f"Strategy overview: {strategy.get('strategy_overview', 'N/A')}"
            )
            self._logger.debug(
                f"LLM raw response (truncated): {strategy.get('raw_response', '')[:200]}..."
            )

        except Exception as e:
            self._logger.error(f"Error creating synthesis plan: {e}", exc_info=True)
            self._synthesis_context = None

    @classmethod
    def from_json(
        cls, filename: str, config: Configuration, **kwargs
    ) -> "MctsSearchTree":
        """
        Create a new search tree by deserialization from a JSON file

        :param filename: the path to the JSON node
        :param config: the configuration of the search
        :return: a deserialized tree
        """
        tree = cls(config, **kwargs)
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        mol_deser = MoleculeDeserializer(dict_["molecules"])
        tree.root = _MODE2NODECLASS[tree.mode].from_dict(
            dict_["tree"], tree, config, mol_deser
        )
        return tree

    def backpropagate(self, from_node: MctsNode) -> None:
        """
        Backpropagate the value estimate and update all nodes from a
        given node all the way to the root.

        :param from_node: the end node of the route to update
        """
        value_estimate = self.compute_reward(from_node)

        current = from_node
        while current is not self.root:
            parent = current.parent
            # For mypy, parent should never by None unless current is the root
            assert parent is not None
            parent.backpropagate(current, value_estimate)  # type: ignore
            current = parent

    def compute_reward(self, node: MctsNode) -> Union[float, Sequence[float]]:
        """
        Compute the reward of a node in the search tree, using one of

            1. Single-objective
            2. Multi-objective
            3. Weighted-sum of multiple specified rewards

        :param node: the node to compute the reward for
        :returns: the value from the scorer(s)
        """
        if self.mode == "single-objective":
            return self.reward_scorer[self.reward_scorer_name](node)

        if self.mode == "multi-objective":
            return self.reward_scorer.score_vector(node)

        return self.reward_scorer.weighted_score(
            node, self.config.search.algorithm_config["search_rewards_weights"]
        )

    def graph(self, recreate: bool = False, from_node=None) -> nx.DiGraph:
        """
        Construct a directed graph object with the nodes as
        vertices and the actions as edges attribute "action".

        :param recreate: if True will construct the graph even though it is cached, defaults to False
        :return: the graph object
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        if not recreate and self._graph:
            return self._graph

        def add_node(node):
            self._graph.add_edge(node.parent, node, action=node.parent[node]["action"])
            for grandchild in node.children:
                add_node(grandchild)

        self._graph = nx.DiGraph()
        # Always add the root
        from_node = from_node or self.root
        self._graph.add_node(from_node)
        for child in from_node.children:
            add_node(child)
        return self._graph

    def nodes(self) -> List[MctsNode]:
        """Return all the nodes in the search tree"""
        return list(self.graph())

    def one_iteration(self) -> bool:
        """
        Perform one iteration of
            1. Selection
            2. Expansion
            3. Rollout
            4. Backpropagation

        :return: if a solution was found
        """
        self.profiling["iterations"] += 1
        leaf = self.select_leaf()
        leaf.expand()

        # A single iteration expands child until it's terminal.
        # So one iteration explores a single path in the tree.
        while not leaf.is_terminal():  # also true if max depth reached
            child = leaf.promising_child()
            if child:
                child.expand()
                leaf = child

        # Try to do intervention here with LLM.

        self.backpropagate(leaf)
        return leaf.state.is_solved

    def select_leaf(self, from_node: Optional[LLMGuidedMctsNode] = None) -> MctsNode:
        """
        Traverse the tree selecting the most promising child at
        each step until leaf node returned.

        :return: the leaf node
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        current = from_node or self.root

        while current.is_expanded and not (
            current.state.is_solved and not self.ignore_stock
        ):
            promising_child = current.promising_child()
            # If promising_child returns None it means that the node
            # is unexpandable, and hence we should break the loop
            if promising_child:
                current = promising_child
        return current

    def serialize(self, filename: str) -> None:
        """
        Serialize the search tree to a JSON file

        :param filename: the path to the JSON file
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        mol_ser = MoleculeSerializer()
        dict_ = {"tree": self.root.serialize(mol_ser), "molecules": mol_ser.store}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=2)

    def _check_mode(self) -> str:
        # if no objective weights are supplied, use multi-objective search
        # if only one objective is specified, search will in be in multi objective mode,
        #  but it is simply a vectoried version of single objective mode
        nrewards = len(self.config.search.algorithm_config["search_rewards"])
        nweights = len(self.config.search.algorithm_config["search_rewards_weights"])

        is_llm_guidance = hasattr(self.config.search, "llm_guidance")

        if is_llm_guidance:
            mode = "llm-guided"
        elif nrewards == 1:
            mode = "single-objective"
        elif nweights == 0:
            mode = "multi-objective"
        else:
            mode = "weighted-sum"

        if mode == "weighted-sum" and nrewards != nweights:
            raise ValueError(
                "number of reward weights are expected to be the same as number of objectives,"
                f"currently have {nweights} weights and {nrewards} objectives)"
            )
        return mode

    def to_image(self, from_node: Optional[MctsNode] = None) -> Any:
        graph = self.graph(recreate=True, from_node=from_node)
        from_node = from_node or self.root
        node_color_map = {}
        for node in graph:
            if node is from_node:
                node_color_map[node] = "cyan"
            elif node.is_solved:
                node_color_map[node] = "green"
            elif node.has_solution:
                node_color_map[node] = "yellow"
            else:
                node_color_map[node] = "red"

        # for node_id, node in enumerate(list(graph)):
        #     labeldict[node] = "{:.02f}".format(get_heuristic_value(node)) + "\n" + "{:.02f}".format(get_value(node))

        pos = graphviz_layout(graph)
        img = nx.draw(
            graph,
            pos,
            with_labels=False,
            node_size=10,
            node_color=[node_color_map[node] for node in graph],
            font_size=8,
            font_color="black",
            font_weight="bold",
            arrows=True,
        )

        return img


class LLMGuidedBeamSearchTree(MctsSearchTree):
    def __init__(
        self,
        config: Configuration,
        root_smiles: Optional[str] = None,
        steer_query: Optional[str] = None,
        previous_attempts: List[PlanningAttempt] = [],
        strategy_json: Optional[str] = None,
    ) -> None:
        super().__init__(
            config, root_smiles, steer_query, previous_attempts, strategy_json
        )
        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
            "iterations": 0,
            "beam_search_steps": 0,
        }

    @property
    def ignore_stock(self):
        return getattr(self.config.expansion_policy, "ignore_stock", False)

    # @weave.op()
    # def one_greedy_iteration(self, stop_at_terminal: bool = True) -> bool:
    #     self.profiling["iterations"] += 1
    #     leaf = self.get_leaf_node()
    #     leaf.expand(stop_at_terminal=stop_at_terminal) # Expand simply add actions to the current node
    #     # leaf.get_heuristic_values()
    #     while leaf._children_actions:
    #         child = leaf.greedy_child()
    #         if child:
    #             child.expand(stop_at_terminal=stop_at_terminal)
    #             leaf = child
    #     return leaf.state.is_solved

    @weave.op()
    async def one_beam_search_step(
        self, candidate_nodes: Optional[List[LLMGuidedMctsNode]] = None
    ) -> List[MctsNode]:
        """
        One step of beam search includes evaluating the current frontier nodes, choose the best ones, then get the actions for the best frontiers.
        """
        self.profiling["beam_search_steps"] += 1
        if candidate_nodes is None:
            candidate_nodes = self.get_all_leaf_nodes(expandable_only=True)
        for node in candidate_nodes:
            if not node.children:
                await node.expand_async()
            # node.get_heuristic_values()
            # frontier_heuristic_values.extend(node._children_heuristic_scores)
            # for child_node_id, action in enumerate(node._children_actions):
            #     candidate_children.append((node, child_node_id))
        candidate_children = []
        frontier_heuristic_values = []

        frontier_heuristic_values_per_node = self.get_route_heuristic_scores(
            candidate_nodes
        )
        for node_id, node in enumerate(candidate_nodes):
            for child_node_id, heuristic_value in enumerate(
                frontier_heuristic_values_per_node[node_id]
            ):
                candidate_children.append((node, child_node_id))
                frontier_heuristic_values.append(heuristic_value)
                # action_str = action_to_str(node._children_actions[child_node_id])
                action_str = node.children[child_node_id].reaction_str
                self._logger.info(
                    f"Score for action {child_node_id} of node {node_id} ({action_str}): {heuristic_value}"
                )

        frontier_heuristic_values = np.array(frontier_heuristic_values, dtype=float)

        beam_width = self.config.search.algorithm_config["W"]
        best_child_node_idx = np.argsort(frontier_heuristic_values)[-beam_width:]

        # if not node.children:
        #     for best_action_id in best_action_idx:
        #         node, child_node_id = candidate_children[best_action_id]
        #         node._instantiate_child(child_node_id)
        next_candidate_nodes = []
        child_keep_idx = defaultdict(list)
        for child_node_id in best_child_node_idx:
            node, child_node_id = candidate_children[child_node_id]
            # If the node has no children, instantiate it
            # if not node.children:
            #     node._instantiate_child(child_node_id)
            next_candidate_nodes.append(node.children[child_node_id])
            child_keep_idx[node].append(child_node_id)

        # remove children nodes that are not selected by beam search
        for node in candidate_nodes:
            # keep_child_idx = child_keep_idx.get(node, [])
            node._purge_children(keep_children=next_candidate_nodes)
            # node.update_children_values_by_heuristic()

        if not getattr(self.config.expansion_policy, "ignore_stock", False):
            next_candidate_nodes = [
                node for node in next_candidate_nodes if node.is_expandable
            ]
        return next_candidate_nodes

    def one_iteration(
        self,
    ) -> bool:
        """
        Perform one iteration of
            1. Selection
            2. Expansion
            3. Rollout
            4. Backpropagation

        :return: if a solution was found
        """
        # self.profiling["iterations"] += 1
        # leaf = self.select_leaf(from_node)

        # max_transforms = max_transforms or self.config.search.algorithm_config.get("max_mcts_transforms", self.config.search.max_transforms)
        # leaf.expand()
        # max_transforms -= 1
        # # A single iteration expands child until it's terminal.
        # # So one iteration explores a single path in the tree.
        # while not leaf.is_terminal() and max_transforms > 0:  # also true if max depth reached
        #     child = leaf.promising_child()
        #     if child:
        #         child.expand()
        #         max_transforms -= 1
        #         leaf = child

        # Try to do intervention here with LLM.
        leaf = self._one_iteration_without_backpropagation()

        self.backpropagate(leaf)
        return leaf.state.is_solved

    def select_leaf(
        self,
        from_node: Optional[LLMGuidedMctsNode] = None,
        max_steps: Optional[int] = None,
    ) -> MctsNode:
        """
        Traverse the tree selecting the most promising child at
        each step until leaf node returned.

        :return: the leaf node
        :raises ValueError: if the tree is not defined
        """
        if not self.root:
            raise ValueError("Root of search tree is not defined ")

        current = from_node or self.root
        while current.is_expanded and not (
            current.state.is_solved and not self.ignore_stock
        ):
            if (
                max_steps is not None
                and len(current.path_to(from_node)[0]) >= max_steps
            ):
                break
            promising_child = current.promising_child()
            # If promising_child returns None it means that the node
            # is unexpandable, and hence we should break the loop
            if promising_child:
                current = promising_child
            else:
                current.is_expandable = False
                break
        return current

    def _get_node_policy(self, node: LLMGuidedMctsNode):
        if node is self.root:
            return None
        return node.path_to()[0][-1].metadata.get("policy_name", None)

    def _one_iteration_without_backpropagation(
        self,
        from_node: Optional[LLMGuidedMctsNode] = None,
        max_steps: Optional[int] = None,
        # max_transforms: Optional[int] = None,
    ) -> LLMGuidedMctsNode:

        self.profiling["iterations"] += 1
        leaf = self.select_leaf(from_node)
        if max_steps is not None and len(leaf.path_to(from_node)[0]) >= max_steps:
            leaf.is_expandable = False
            return leaf
        leaf.expand()
        # max_transforms = max_transforms or self.config.search.algorithm_config.get("max_mcts_transforms", self.config.search.max_transforms)
        # leaf.expand()
        # max_transforms -= 1
        # A single iteration expands child until it's terminal.
        # So one iteration explores a single path in the tree.
        while (
            self.ignore_stock or not leaf.is_terminal()
        ):  # if ignore_stock is True, is_terminal is ignored
            # while not (
            #     leaf.state.max_transforms >= self.config.search.max_transforms or \
            #     leaf.state.is_solved and not self.ignore_stock or \
            #     not leaf.is_expandable
            # ):
            if max_steps is not None and len(leaf.path_to(from_node)[0]) >= max_steps:
                leaf.is_expandable = False
                break
            # if max_transforms is not None and leaf.state.max_transforms >= max_transforms:
            #     break
            child = leaf.promising_child()
            if child:
                child.expand()
                # max_transforms -= 1
                leaf = child
            else:
                break

        return leaf

    def backpropagate(
        self, from_node: MctsNode, to_node: Optional[MctsNode] = None
    ) -> None:
        """
        Backpropagate the value estimate and update all nodes from a
        given node all the way to the root.

        :param from_node: the end node of the route to update
        """
        value_estimate = self.compute_reward(from_node)

        current = from_node
        while current is not self.root:
            if to_node and current is to_node:
                break
            parent = current.parent
            # For mypy, parent should never by None unless current is the root
            assert parent is not None
            parent.backpropagate(current, value_estimate)  # type: ignore
            current = parent

    # def _get_current_beam_search_candidate_nodes(self):
    #     def _is_beam_search_candidate(node):
    #         if node.is_solved:
    #             return False
    #         if sum(node._children_heuristic_scores) > 0.0:
    #             return False
    #         # if None in node._children: # None in children means that a child of this node has been purged, which means this node has been expanded and validated.
    #         #     return False
    #         return True

    #     res = []
    #     for node in self.graph(recreate=True):
    #         if _is_beam_search_candidate(node) and node.parent not in res:
    #             res.append(node)
    #     return res

    @property
    def current_beam_search_step(self):
        """Get the current beam search step based on the maximum length of the smiles tree"""
        # candidate_nodes = self._get_current_beam_search_candidate_nodes()
        candidate_nodes = self.get_all_leaf_nodes()
        if not candidate_nodes:
            return 0
        return max([len(node.get_smiles_tree()) for node in candidate_nodes])

    @weave.op()
    async def beam_search(
        self, step_limit: Optional[int] = None, policy: str = "llm_guided"
    ):
        self.config.expansion_policy.selection = [policy]

        step_limit = (
            step_limit
            if step_limit is not None
            else self.config.search.llm_guidance.step_limit
        )
        # candidate_node = [self.root]
        # candidate_nodes = self._get_current_beam_search_candidate_nodes()
        candidate_nodes = self.get_all_leaf_nodes()
        current_step = self.current_beam_search_step
        while (
            candidate_nodes
        ):  # The search would stop if all leaf nodes are solved or don't have _children_actions
            current_step += 1
            if step_limit and current_step > step_limit:
                self._logger.info(
                    f"Beam search reached step limit of {step_limit}. Stopping."
                )
                break
            self._logger.info(f"Beam search step {current_step}")
            candidate_nodes = await self.one_beam_search_step(candidate_nodes)

        return self.get_all_leaf_nodes(include_expanded=True)

    @weave.op()
    def mcts(
        self,
        iteration_limit: Optional[int] = None,
        max_transforms: Optional[int] = None,
        max_steps: Optional[int] = None,
        policy: str = "multi_expansion_strategy",
        max_llm_guided_depth: Optional[int] = None,
        preprocess: bool = True,
    ):
        self.config.expansion_policy.selection = [policy]

        iteration_limit = (
            iteration_limit
            if iteration_limit is not None
            else self.config.search.iteration_limit
        )

        if preprocess:
            self._search_post_processing(max_llm_guided_depth)

        for _ in range(iteration_limit):
            # self.one_iteration(max_transforms)
            leaf = self._one_iteration_without_backpropagation(
                max_steps=max_steps,
            )
            self.backpropagate(leaf)

    @weave.op()
    def nested_mcts(
        self,
        iteration_limit: Optional[int] = None,
        child_iteration_limit: Optional[int] = None,
        # max_transforms: Optional[int]=None,
        max_steps: Optional[int] = None,
        # max_child_transforms: Optional[int]=None,
        max_child_steps: Optional[int] = None,
        policy: str = "llm_query_explorer",
        child_policy: str = "multi_expansion_strategy",
        max_llm_guided_depth: Optional[int] = None,
        preprocess: bool = True,
    ):

        """
        After each iteration of the main MCTS, a child MCTS is run from the leaf node.
        The child MCTS will not run if the leaf node is solved, or if the leaf node has not reached the max_transforms or max_depth

        TODO: This function is deprecated. Use mcts instead.
        """

        # max_transforms = max_transforms or self.config.search.max_transforms

        iteration_limit = (
            iteration_limit
            if iteration_limit is not None
            else self.config.search.iteration_limit
        )

        child_iteration_limit = (
            child_iteration_limit
            if child_iteration_limit is not None
            else self.config.search.child_iteration_limit
        )

        if preprocess:
            self._search_post_processing(max_llm_guided_depth)

        for _ in range(iteration_limit):
            self.config.expansion_policy.selection = [policy]
            leaf = self._one_iteration_without_backpropagation(
                # max_transforms=max_transforms,
                max_steps=max_steps
            )
            if leaf.state.is_solved:
                self.backpropagate(leaf)
                continue

            # if max_transforms is not None and leaf.state.max_transforms < max_transforms:
            #     self.backpropagate(leaf)
            #     continue

            if max_steps is not None and len(leaf.path_to()[0]) != max_steps:
                # it is mandatory that all steps of LLM guided queries are found
                self.backpropagate(leaf)
                continue

            leaf.reset_expansion()

            if child_iteration_limit:
                self.config.expansion_policy.selection = [child_policy]
                for _ in range(child_iteration_limit):
                    child_leaf = self._one_iteration_without_backpropagation(
                        from_node=leaf,
                        # max_transforms=max_child_transforms,
                        max_steps=max_child_steps,
                    )
                    if (
                        child_leaf.state.is_solved
                    ):  # stop once a solution of the child search is found. Backpropagate to the root.
                        self.backpropagate(child_leaf)
                        break
                    else:
                        self.backpropagate(child_leaf, to_node=leaf)

                    # if len(child_leaf.path_to(leaf)[0]) >= max_child_steps:
                    #     break

            else:
                self.backpropagate(leaf)
            # self.backpropagate(leaf)

    def get_leaf_node(self):
        """Each node is expected to have a single child, although there could be more than one actions"""
        current = self.root
        while current.children:
            current = current.children[0]
        return current

    def get_all_leaf_nodes(
        self, include_expanded: bool = False, expandable_only: bool = False
    ) -> List[LLMGuidedMctsNode]:
        candidate_nodes = []
        for node in list(self.graph(recreate=True)):
            if not any(node.children):
                candidate_nodes.append(node)
        if not include_expanded:
            candidate_nodes = [node for node in candidate_nodes if not node.is_expanded]
        if expandable_only:
            candidate_nodes = [node for node in candidate_nodes if node.is_expandable]
        return candidate_nodes

    def get_route_heuristic_scores(self, nodes):
        heuristic_score_fn = get_route_heuristic_score_fn(self.config)
        return heuristic_score_fn(nodes)

    def prune_llm_guided_error_nodes(self):
        for node in self.get_all_leaf_nodes(include_expanded=True):
            parent = node.parent
            if (
                parent
                and parent.children
                and len(parent._children) != len(parent._children_heuristic_scores)
            ):
                parent.reset_expansion()

    def prune_by_max_depth(self, max_depth: int):
        node_depths = [node.calculate_path_depth() for node in self.nodes()]
        found_max_depth = max(node_depths)
        max_depth = min(max_depth, found_max_depth)

        for node in self.nodes():
            if node.calculate_path_depth() >= max_depth:
                node.reset_expansion()

        for node in self.get_all_leaf_nodes(include_expanded=True):
            if node.calculate_path_depth() < max_depth:
                node.is_expandable = False

    def prune_mono_branches(self):
        def _get_prune_point(node):
            current = node
            if current.is_solved:
                return current
            while current.parent and len(current.parent.children) == 1:
                current = current.parent
            return current

        for node in self.get_all_leaf_nodes(include_expanded=True):
            _get_prune_point(node).reset_expansion()

    def compute_reward(self, node):
        return self.reward_scorer[self.reward_scorer_name](node)

    def _search_post_processing(self, max_llm_guided_depth: Optional[int] = None):
        """
        Post-processing of the beam search results.
        This can include pruning, filtering, or any other operations needed
        after the beam search is complete.
        """
        self._logger.info("Post-processing beam search results.")
        self.prune_llm_guided_error_nodes()
        if max_llm_guided_depth is None:
            max_llm_guided_depth = self.config.search.llm_guidance.step_limit
        self.prune_by_max_depth(max_llm_guided_depth)

        # if self.config.search.algorithm_config.get("prune_mono_branches", True):
        #     self.prune_mono_branches()

        self.graph(recreate=True)  # Recreate the graph to reflect the changes

        for node in self.nodes():
            if node.is_expanded:
                node.initialize_children_values(from_heuristic=True)

        # for node in self.get_all_leaf_nodes(include_expanded=True):
        #     if node.is_terminal():
        #         self.backpropagate(node)

    def reset_node_values(self):
        for node in self.nodes():
            if node.is_expanded:
                node.initialize_children_values(from_heuristic=False)

    def get_solved_routes(self):
        routes = []
        for node in self.get_all_leaf_nodes(include_expanded=True):
            if node.is_solved:
                route = node.to_reaction_tree()
                routes.append(route)

        routes = deduplicate_routes(routes)
        return routes

    def save_solved_routes(self, json_path: str):
        routes = self.get_solved_routes()
        routes = [each.to_dict() for each in routes]

        with open(json_path, "w") as f:
            json.dump(routes, f, indent=2)

        self._logger.info(f"{len(routes)} solved routes are saved to: {json_path}")
