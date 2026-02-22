import asyncio
import json
import logging
import os
from typing import List, Optional
from synthelite.agent_search.utils import get_step_description, summarize_tree
from synthelite.aizynthfinder import AiZynthFinder
from synthelite.analysis.routes import RouteCollection
from synthelite.chem.mol import Molecule
from synthelite.context.config import Configuration
from synthelite.search.mcts.search import LLMGuidedBeamSearchTree
from synthelite.utils.clogging import init_logger
from synthelite.utils.exceptions import MoleculeException
from synthelite.utils.reaction_tree import deduplicate_routes
from synthelite.utils.type_utils import StrDict

import weave


def _deduplicate_query_sets(query_sets: list[tuple[str]]):
    seen = set()
    unique_query_sets = []
    for query_set in query_sets:
        if query_set not in seen:
            seen.add(query_set)
            unique_query_sets.append(query_set)
    return unique_query_sets


class SyntheliteLLMFinder(AiZynthFinder):

    log_file = "synthelite_llm_finder.log"
    weave_url_file = "weave_url.txt"
    llm_guided_tree_file = "tree.{}.llm_guided.json"
    llm_guided_route_file = "routes.llm_guided.json"

    query_explored_tree_file = "tree.{}.llm_query_explorer.json"
    query_explored_route_file = "routes.llm_query_explorer.json"

    mcts_tree_file = "tree.{}.mcts.json"
    mcts_route_file = "routes.mcts.json"

    mcts_suffix = "mcts.json"

    _default_steer_query = "Highly feasible synthesis with high overall yields, consider potential side reactions and byproducts. Also ensure no unnecesary reactions are performed."

    def __init__(
        self, configfile: Optional[str] = None, configdict: Optional[StrDict] = None
    ) -> None:
        super().__init__(configfile=configfile, configdict=configdict)
        self.stock.select_all()
        self._steer_query: Optional[str]

        self._save_dir: Optional[str] = None

        self._previous_attempt_cache = {}

        self._n_llm_trees = 0
        self._n_explore_trees = 0

        self._logger = init_logger()

    def setup_finder(
        self,
        target_smiles: str,
        steer_query: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        self.target_smiles = target_smiles
        self.steer_query = steer_query
        # if save_dir is not None:
        #     self.save_dir = save_dir
        # elif self._save_dir is None:
        self.save_dir = os.getcwd() if save_dir is None else save_dir

    @property
    def steer_query(self) -> str:
        """The query to guide the search."""
        if self._steer_query is None:
            return self._default_steer_query
        return self._steer_query

    @steer_query.setter
    def steer_query(self, query: str) -> None:
        """Set the steering query for the search."""
        self._steer_query = query

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @save_dir.setter
    def save_dir(self, path: str) -> None:
        """Set the directory to save results."""
        if not os.path.exists(path):
            os.makedirs(path)
        # self._logger = init_logger(os.path.join(path, self.log_file))
        self._save_dir = path

    @property
    def tree_llm_guided_path(self) -> str:
        return os.path.join(self.save_dir, self.llm_guided_tree_file)

    @property
    def route_llm_guided_path(self) -> str:
        return os.path.join(self.save_dir, self.llm_guided_route_file)

    @property
    def query_explored_tree_path(self) -> str:
        return os.path.join(self.save_dir, self.query_explored_tree_file)

    @property
    def query_explored_route_path(self) -> str:
        return os.path.join(self.save_dir, self.query_explored_route_file)

    @property
    def mcts_tree_path(self) -> str:
        return os.path.join(self.save_dir, self.mcts_tree_file)

    @property
    def mcts_route_path(self) -> str:
        return os.path.join(self.save_dir, self.mcts_route_file)

    def prepare_search(self, tree_path: str = None) -> None:
        """
        Setup the tree for searching

        :raises ValueError: if the target molecule was not set
        """
        if not self.target_mol:
            raise ValueError("No target molecule set")

        try:
            self.target_mol.sanitize()
        except MoleculeException:
            raise ValueError("Target molecule unsanitizable")

        self.stock.select_all()
        self.stock.reset_exclusion_list()
        if (
            self.config.search.exclude_target_from_stock
            and self.target_mol in self.stock
        ):
            self.stock.exclude(self.target_mol)
            self._logger.debug("Excluding the target compound from the stock")

        if self.config.search.break_bonds or self.config.search.freeze_bonds:
            self._setup_focussed_bonds(self.target_mol)

        # self._setup_search_tree(tree_path=tree_path)
        self.analysis = None
        self.routes = RouteCollection([])
        self.filter_policy.reset_cache()
        self.expansion_policy.reset_cache()

    def _setup_search_tree(
        self, tree_path: str = None, previous_attempts: List[dict] = []
    ) -> None:
        self._logger.debug(f"Defining tree root:  {self.target_smiles}")
        if tree_path is not None and os.path.exists(tree_path):
            self.tree = LLMGuidedBeamSearchTree.from_json(
                tree_path,
                self.config,
                root_smiles=self.target_smiles,
                steer_query=self.steer_query,
                previous_attempts=previous_attempts,
            )
            self._logger.info(f"Search tree loaded from {tree_path}")
        else:
            self.tree = LLMGuidedBeamSearchTree(
                root_smiles=self.target_smiles,
                steer_query=self.steer_query,
                previous_attempts=previous_attempts,
                config=self.config,
            )
        # Set finder reference for access to synthesis_plan in nodes
        self.tree._finder = self
        self.filter_policy.reset_cache()
        self.expansion_policy.reset_cache()

    @weave.op()
    async def _set_up_search_tree_from_id(
        self, tree_id: int, with_previous_attempts: bool = False
    ):
        tree_path = self.tree_llm_guided_path.format(tree_id)
        previous_attempts = []
        if with_previous_attempts:
            for i in range(tree_id):
                previous_tree_path = self.tree_llm_guided_path.format(i)
                if previous_tree_path in self._previous_attempt_cache:
                    previous_attempts.append(
                        self._previous_attempt_cache[previous_tree_path]
                    )
                    continue
                previous_tree = LLMGuidedBeamSearchTree.from_json(
                    previous_tree_path,
                    self.config,
                    root_smiles=self.target_smiles,
                    steer_query=self.steer_query,
                )
                summary_route = summarize_tree(previous_tree, attempt_id=str(i))
                if self.config.route_validator:
                    summary_route = await self.config.route_validator.validate_route(
                        summary_route,
                        target_smiles=self.target_smiles,
                        user_constraint=self.steer_query,
                    )
                else:
                    summary_route.feedback = "Try a different approach while still adhering to the user constraint."

                self._previous_attempt_cache[
                    previous_tree_path
                ] = summary_route.to_dict()
                previous_attempts.append(summary_route.to_dict())
        self._setup_search_tree(
            tree_path=tree_path, previous_attempts=previous_attempts
        )

    @weave.op()
    async def beam_search(
        self,
        attempt_number: Optional[int] = None,
        gather_routes: bool = True,
    ) -> None:
        self.expansion_policy.selection = ["llm_guided"]
        # n_runs = self.config.search.llm_guidance.n_runs

        with weave.attributes(
            {"target": self.target_smiles, "query": self.steer_query}
        ):
            all_routes = []
            # for n in range(n_runs):
            n = 0 if attempt_number is None else attempt_number
            tree_llm_guided_path = self.tree_llm_guided_path.format(n)
            try:
                await self._set_up_search_tree_from_id(
                    tree_id=n, with_previous_attempts=True
                )
                _, call = await self.tree.beam_search.call(self.tree)
                if not os.path.exists(tree_llm_guided_path):
                    if call.exception:
                        self._logger.error(
                            f"Beam search failed with exception: {call.exception}",
                            exc_info=True,
                        )
                        self._logger.error("Saving partial results...")

                    url = f"https://wandb.ai/{call.project_id}/r/call/{call.id}"
                    self._logger.info(f"See LLM traces at: {url}")
                    with open(
                        os.path.join(self.save_dir, self.weave_url_file), "a"
                    ) as f:
                        f.write(url)
                        f.write("\n")
                    self.tree.serialize(tree_llm_guided_path)

            except Exception as e:
                self._logger.error(
                    f"Beam search failed with exception: {e}", exc_info=True
                )
                self._logger.error("Saving partial results...")

            self._logger.info(
                f"Search tree saved to {self.tree_llm_guided_path.format(n)}"
            )
            if gather_routes:
                # self.tree.save_solved_routes(self.route_llm_guided_path)
                all_routes.extend(self.tree.get_solved_routes())

                all_routes = deduplicate_routes(all_routes)
                all_routes_dict = [route.to_dict() for route in all_routes]

                with open(self.route_llm_guided_path, "w") as f:
                    json.dump(all_routes_dict, f, indent=2)
                self._logger.info(
                    f"Solved routes saved to {self.route_llm_guided_path}"
                )

    def mcts_from_beam_search(
        self,
        prune_beam_search_steps: Optional[List[int]] = None,
        policy: str = "multi_expansion_strategy",
    ):
        if prune_beam_search_steps is None:
            prune_beam_search_steps = [prune_beam_search_steps]
        for (
            beam_search_max_depth
        ) in prune_beam_search_steps:  # Should I put this to args or config?
            if beam_search_max_depth is not None:
                tree_save_path_mcts = os.path.join(
                    self.save_dir, f"tree.depth_{beam_search_max_depth}.{policy}.json"
                )
                route_save_path_mcts = os.path.join(
                    self.save_dir, f"routes.depth_{beam_search_max_depth}.{policy}.json"
                )
            else:
                tree_save_path_mcts = os.path.join(self.save_dir, f"tree.{policy}.json")
                route_save_path_mcts = os.path.join(
                    self.save_dir, f"routes.{policy}.json"
                )

            if os.path.exists(tree_save_path_mcts):
                self._logger.info(
                    f"Search tree for MCTS already exists at {tree_save_path_mcts}, skipping MCTS."
                )
                continue

            self.tree = LLMGuidedBeamSearchTree.from_json(
                self.tree_llm_guided_path,
                self.config,
                root_smiles=self.target_smiles,
                steer_query=self.steer_query,
            )

            self._logger.info(
                f"Loaded search tree for MCTS from {self.tree_llm_guided_path} and prune to max depth {beam_search_max_depth}"
            )

            self.tree.mcts(
                max_llm_guided_depth=beam_search_max_depth,
                policy=policy,
            )
            self.tree.serialize(tree_save_path_mcts)
            self._logger.info(f"MCTS search tree saved to {tree_save_path_mcts}")

            self.tree.save_solved_routes(route_save_path_mcts)

    def mcts_from_query_explore(self, skip_if_exist: bool = True):
        if self._n_explore_trees == 0:
            self._logger.info(
                "No LLM query explored trees found, skipping MCTS from query explore."
            )
            return

        # if skip_if_exist and os.path.exists(self.mcts_route_path):
        #     self._logger.info(f"MCTS routes already exist at {self.mcts_route_path}, skipping MCTS from query explore.")
        #     return

        solved_routes = []
        for i in range(self._n_explore_trees):
            tree_path = self.query_explored_tree_path.format(i)
            # mcts_tree_path = os.path.join(self.save_dir, f'tree.{i}.{self.mcts_suffix}')
            mcts_tree_path = self.mcts_tree_path.format(i)

            if skip_if_exist and os.path.exists(mcts_tree_path):
                self._logger.info(
                    f"Search tree for MCTS already exists at {mcts_tree_path}, skipping MCTS."
                )
                tree = LLMGuidedBeamSearchTree.from_json(
                    mcts_tree_path,
                    self.config,
                    root_smiles=self.target_smiles,
                    steer_query=self.steer_query,
                )
            else:
                tree = LLMGuidedBeamSearchTree.from_json(
                    tree_path,
                    self.config,
                    root_smiles=self.target_smiles,
                    steer_query=self.steer_query,
                )
                tree.reset_node_values()

                self._logger.info(f"Loaded search tree for MCTS from {tree_path}")

                tree.mcts(
                    policy="multi_expansion_strategy",
                )
                tree.serialize(mcts_tree_path)
                self._logger.info(f"MCTS search tree saved to {mcts_tree_path}")

            solved_routes.extend(tree.get_solved_routes())
            # tree.save_solved_routes(mcts_route_path)
            # self._logger.info(f"MCTS solved routes saved to {mcts_route_path}")
        solved_routes = deduplicate_routes(solved_routes)
        with open(self.mcts_route_path, "w") as f:
            json.dump([route.to_dict() for route in solved_routes], f, indent=2)

    def load_tree(self, tree_path: Optional[str]):
        if tree_path is not None:
            tree = LLMGuidedBeamSearchTree.from_json(
                tree_path,
                self.config,
                root_smiles=self.target_smiles,
                steer_query=self.steer_query,
            )
            tree._search_post_processing()
        else:
            tree = LLMGuidedBeamSearchTree(
                config=self.config,
                root_smiles=self.target_smiles,
                steer_query=self.steer_query,
            )
        tree._finder = self
        self.filter_policy.reset_cache()
        self.expansion_policy.reset_cache()
        return tree

    # def _get_llm_query(self, action):
    #     if 'lmdata' in action.metadata:
    #         return action.metadata.get('lmdata', {}).get('query', '')
    #     elif 'description' in action.metadata:
    #         return action.metadata.get('description', '')
    #     elif 'template_code' in action.metadata:
    #         template_code = int(action.metadata["template_code"])
    #         template_df = self.config.template_search.df
    #         template_row = template_df[template_df['template_code'] == template_code]
    #         if not template_row.empty:
    #             return template_row.llm_description.item()

    #     else:
    #         return action.metadata.get('query', '')

    def _get_queries_and_queried_actions_from_leaf(self, node):
        action_path, _ = node.path_to()
        if not action_path:
            return tuple(), tuple()
        # if 'lmdata' in action_path[0].metadata:
        #     queries = [each.metadata.get('lmdata', {}).get('query', '') for each in action_path]
        # else:
        #     queries = [each.metadata.get('query', '') for each in action_path]
        # queries = [each for each in queries if each]
        queries = []
        actions = []
        for action in action_path:
            # query = self._get_llm_query(action)
            query = get_step_description(
                action, self.config.template_search.df, use_llm_query=False
            )
            if query:
                queries.append(query)
                actions.append(action)
        return tuple(queries), tuple(actions)

    def _get_llm_queries_and_ref_actions(
        self,
        # tree_ids: Optional[list[int]]=None,
        llm_tree_paths: Optional[List[str]] = None,
    ) -> List[tuple[str]]:
        # self.tree._search_post_processing()
        # leafs = self.tree.get_all_leaf_nodes()
        n_trees = self.config.search.llm_guidance.n_runs

        if not llm_tree_paths:
            tree_ids = list(range(n_trees))
            llm_tree_paths = [self.tree_llm_guided_path.format(n) for n in tree_ids]
        # tree_ids = tree_ids if tree_ids is not None else list(range(n_trees))

        query_sets, ref_action_sets = [], []

        seen_query_sets = set()

        for llm_tree_path in llm_tree_paths:
            # llm_tree_path = self.tree_llm_guided_path.format(n)
            if not os.path.exists(llm_tree_path):
                self._logger.warning(
                    f"LLM guided tree file {llm_tree_path} does not exist, skipping."
                )
                continue
            llm_tree = self.load_tree(llm_tree_path)
            leafs = llm_tree.get_all_leaf_nodes()

            for leaf in leafs:
                (
                    query_sets_i,
                    ref_actions_i,
                ) = self._get_queries_and_queried_actions_from_leaf(leaf)
                if query_sets_i and query_sets_i not in seen_query_sets:
                    seen_query_sets.add(query_sets_i)
                    query_sets.append(query_sets_i)
                    ref_action_sets.append(ref_actions_i)

        # query_sets = [_get_queries_and_queried_actions_from_leaf(leaf)[0] for leaf in leafs]
        # query_sets = _deduplicate_query_sets(query_sets)

        return query_sets, ref_action_sets

    def query_explore_search(
        self,
        llm_tree_paths: List[str],
        tree_save_path: Optional[str] = None,
        skip_if_exists: bool = True,
        policy: str = "llm_query_explorer",
        gather_routes: bool = True,
    ) -> None:

        if "llm_query_explorer" not in self.config.expansion_policy._items:
            self._logger("LLM query explorer not configured, skipping _query_explore.")
            return

        self._logger.info("Starting LLM query exploration...")
        # self.expansion_policy.selection = ["llm_query_explorer"]

        query_sets, ref_actions_sets = self._get_llm_queries_and_ref_actions(
            llm_tree_paths
        )

        # explore_forest = []
        solved_routes = []
        for i, (queries, ref_actions) in enumerate(zip(query_sets, ref_actions_sets)):
            tree_path = (
                self.query_explored_tree_path.format(i)
                if tree_save_path is None
                else tree_save_path
            )
            if skip_if_exists and os.path.exists(tree_path):
                self._logger.info(
                    f"LLM query exploration tree already exists at {tree_path}, skipping."
                )
                # explore_tree = LLMGuidedBeamSearchTree.from_json(
                #     tree_path,
                #     self.config,
                #     root_smiles=self.target_smiles,
                #     steer_query=self.steer_query,
                # )
                explore_tree = self.load_tree(tree_path)

            else:
                self.config.expansion_policy["llm_query_explorer"].queries = queries
                self.config.expansion_policy[
                    "llm_query_explorer"
                ].ref_actions = ref_actions

                explore_tree = self.load_tree(tree_path=None)
                print("Searching with queries:", queries)

                # explore_tree.nested_mcts(
                #     iteration_limit=self.config.search.query_explore_iteration_limit,
                #     child_iteration_limit=self.config.search.neural_guided_iteration_limit,
                #     max_steps=len(queries),
                #     max_child_steps=3,
                #     policy=policy,
                #     child_policy='multi_expansion_strategy',
                # )
                explore_tree.mcts(
                    iteration_limit=self.config.search.query_explore_iteration_limit,
                    max_steps=len(queries),
                    policy=policy,
                    # max_llm_guided_depth=len(queries),
                    preprocess=False,
                )

                explore_tree.serialize(tree_path)

                self._logger.info(f"LLM query exploration tree saved to {tree_path}")

            solved_routes.extend(explore_tree.get_solved_routes())

        self._n_explore_trees = len(query_sets)

        if gather_routes:
            solved_routes = deduplicate_routes(solved_routes)
            # route_file = f'routes.llm_query_explorer.json'
            with open(self.query_explored_route_path, "w") as f:
                json.dump([route.to_dict() for route in solved_routes], f, indent=2)

    # def query_explore_with_neural_guided(
    #     self,
    # ):
    #     query_sets = self._get_llm_queries()
    # @weave.op()
    def gather_routes(self, tree_paths: List[str], route_save_path: str) -> None:
        """Routes should be gathered in reversed because later iterations are supposed to be better."""
        n_routes_per_tree = self.config.post_processing.max_routes // len(tree_paths)
        all_routes = []

        for tree_path in tree_paths:
            if not os.path.exists(tree_path):
                self._logger.warning(f"Tree file {tree_path} does not exist, skipping.")
                continue
            tree = self.load_tree(tree_path)
            routes = tree.get_solved_routes()
            routes = sorted(
                routes, key=lambda x: len(list(x.reactions())), reverse=True
            )  # prioritize longer routes
            routes = routes[:n_routes_per_tree]
            for route in routes:
                route.metadata["tree_path"] = tree_path
            all_routes.extend(routes)

        all_routes = deduplicate_routes(all_routes)
        all_routes_dict = [route.to_dict(include_metadata=True) for route in all_routes]

        with open(route_save_path, "w") as f:
            json.dump(all_routes_dict, f, indent=2)

    async def tree_search(
        self,
        skip_tree_if_exist: bool = False,
        # prune_beam_search_steps: Optional[List[int]] = None
    ) -> None:
        """
        Beam search with LLMs is skipped if `skip_llm_guided` is True and the tree file already exists.
        """
        # if not self.tree:
        self.prepare_search(self.tree_llm_guided_path)
        # if not (skip_llm_guided and os.path.exists(self.tree_llm_guided_path.format(0))):
        #     await self.beam_search()
        # self.query_explore_search(policy='llm_query_explorer')
        n_runs = self.config.search.llm_guidance.n_runs

        for n in range(n_runs):
            if skip_tree_if_exist and os.path.exists(
                self.tree_llm_guided_path.format(n)
            ):
                self._logger.info(
                    f"LLM-guided tree already exists at {self.tree_llm_guided_path.format(n)}, skipping LLM guided beam search and query exploration."
                )
            else:
                await self.beam_search(attempt_number=n, gather_routes=False)

            if skip_tree_if_exist and os.path.exists(
                self.query_explored_tree_path.format(n)
            ):
                self._logger.info(
                    f"LLM query explored tree already exists at {self.query_explored_tree_path.format(n)}, skipping query exploration."
                )

            else:
                self.query_explore_search(
                    llm_tree_paths=[self.tree_llm_guided_path.format(n)],
                    tree_save_path=self.query_explored_tree_path.format(n),
                    gather_routes=False
                    # policy='llm_query_explorer',
                )

        self.gather_routes(
            tree_paths=[
                self.query_explored_tree_path.format(i)
                for i in range(n_runs - 1, -1, -1)
            ],
            route_save_path=self.query_explored_route_path,
        )
