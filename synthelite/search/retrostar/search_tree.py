""" Module containing a class that holds the tree search
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import asyncio

from synthelite.chem.serialization import MoleculeDeserializer, MoleculeSerializer
from synthelite.search.andor_trees import AndOrSearchTreeBase, SplitAndOrTree
from synthelite.search.retrostar.cost import MoleculeCost, ReactionCost
from synthelite.search.retrostar.nodes import MoleculeNode
from synthelite.utils.exceptions import RejectionException
from synthelite.utils.clogging import logger

if TYPE_CHECKING:
    from synthelite.chem import RetroReaction
    from synthelite.context.config import Configuration
    from synthelite.reactiontree import ReactionTree
    from synthelite.utils.type_utils import List, Optional, Sequence, Tuple


class SearchTree(AndOrSearchTreeBase):
    """
    Encapsulation of the Retro* search tree (an AND/OR tree).

    :ivar config: settings of the tree search algorithm
    :ivar root: the root node

    :param config: settings of the tree search algorithm
    :param root_smiles: the root will be set to a node representing this molecule, defaults to None
    """

    def __init__(
        self, config: Configuration, root_smiles: Optional[str] = None
    ) -> None:
        super().__init__(config, root_smiles)
        self._mol_nodes: List[MoleculeNode] = []
        self._logger = logger()
        self.molecule_cost = MoleculeCost(config)

        if root_smiles:
            self.root: Optional[MoleculeNode] = MoleculeNode.create_root(
                root_smiles, config, self.molecule_cost
            )
            self._mol_nodes.append(self.root)
        else:
            self.root = None

        self._routes: List[ReactionTree] = []

        self.profiling = {
            "expansion_calls": 0,
            "reactants_generations": 0,
        }

        self.reaction_cost = ReactionCost(config)

        self.count_rxns = 0
        self.rxns = set()

    @classmethod
    def from_json(cls, filename: str, config: Configuration) -> SearchTree:
        """
        Create a new search tree by deserialization from a JSON file

        :param filename: the path to the JSON node
        :param config: the configuration of the search tree
        :return: a deserialized tree
        """

        def _find_mol_nodes(node):
            for child_ in node.children:
                tree._mol_nodes.append(child_)  # pylint: disable=protected-access
                for grandchild in child_.children:
                    _find_mol_nodes(grandchild)

        tree = cls(config)
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)
        mol_deser = MoleculeDeserializer(dict_["molecules"])
        tree.root = MoleculeNode.from_dict(
            dict_["tree"], config, mol_deser, tree.molecule_cost
        )
        tree._mol_nodes.append(tree.root)  # pylint: disable=protected-access
        for child in tree.root.children:
            _find_mol_nodes(child)
        return tree

    @property
    def mol_nodes(self) -> Sequence[MoleculeNode]:  # type: ignore
        """Return the molecule nodes of the tree"""
        return self._mol_nodes

    def one_iteration(self) -> bool:
        """
        Perform one iteration of
            1. Selection
            2. Expansion
            3. Update

        :raises StopIteration: if the search should be pre-maturely terminated
        :return: if a solution was found
        :rtype: bool
        """
        if self.root is None:
            raise ValueError("Root is undefined. Cannot make an iteration")

        self._routes = []

        next_node = self._select()

        if not next_node:
            self._logger.debug("No expandable nodes in Retro* iteration")
            raise StopIteration

        self._expand(next_node)

        if not next_node.children:
            next_node.expandable = False

        self._update(next_node)
        # print(self.molecule_cost.count)
        # print(len(self.molecule_cost.mols))

        return self.root.solved

    def routes(self) -> List[ReactionTree]:
        """
        Extracts and returns routes from the AND/OR tree

        :return: the routes
        """
        if self.root is None:
            return []
        if not self._routes:
            self._routes = SplitAndOrTree(self.root, self.config.stock).routes
        return self._routes

    def serialize(self, filename: str) -> None:
        """
        Seralize the search tree to a JSON file

        :param filename: the path to the JSON file
        :type filename: str
        """
        if self.root is None:
            raise ValueError("Cannot serialize tree as root is not defined")

        mol_ser = MoleculeSerializer()
        dict_ = {"tree": self.root.serialize(mol_ser), "molecules": mol_ser.store}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=2)

    async def run_llm_cost(self, reactions: List[RetroReaction]) -> float:
        return await asyncio.gather(
            *[
                self.reaction_cost(r.reversed_mapped_reaction_smiles())
                for r in reactions
            ]
        )

    def rank_llm(self, reactions: List[RetroReaction]) -> Tuple[List[float], List[str]]:
        """Make llm call to rank reactions"""
        responses = asyncio.run(self.run_llm_cost(reactions))
        llmcost = [
            self.reaction_cost.reaction_cost._parse_score(response)
            for response in responses
        ]
        return llmcost, responses

    def rerank(
        self, reactions: List[RetroReaction], cost: List[float], rrcost: List[float]
    ) -> List[List[RetroReaction], List[float]]:
        """Dummiest reranking: multiply cost and llm cost"""
        newcost = [c * r for c, r in zip(cost, rrcost)]
        return [
            r for _, r in sorted(zip(newcost, reactions), key=lambda x: x[0])
        ], sorted(newcost)

    def _expand(self, node: MoleculeNode) -> None:
        reactions, priors = self.config.expansion_policy([node.mol])
        self.profiling["expansion_calls"] += 1

        if not reactions:
            return

        costs = -np.log(np.clip(priors, 1e-3, 1.0))

        reactions_to_expand = []
        reaction_costs = []
        for reaction, cost in zip(reactions, costs):
            try:
                self.profiling["reactants_generations"] += 1
                _ = reaction.reactants
            except:  # pylint: disable=bare-except
                continue
            if not reaction.reactants:
                continue
            for idx, _ in enumerate(reaction.reactants):
                rxn_copy = reaction.copy(idx)
                if self._filter_reaction(rxn_copy):
                    continue
                reactions_to_expand.append(rxn_copy)
                reaction_costs.append(cost)

        self.count_rxns += len(costs)
        self.rxns.update(
            [
                r.metadata["template_hash"] + ":::" + r.mol.smiles
                for r in reactions_to_expand
            ]
        )  # specify reaction smiles (object contains also cost which is updated)

        # Rerank rxns with llm
        # llmcost, raw_responses = self.rank_llm(reactions_to_expand)
        # new_reactions, new_costs = self.rerank(reactions_to_expand, reaction_costs, llmcost)

        # for cost, rxn in zip(reaction_costs, reactions_to_expand):
        # for cost, rxn in zip(new_costs, new_reactions):
        for cost, rxn in zip(costs, reactions_to_expand):
            rxn.metadata["cost"] = cost
            # rxn.metadata["llm_cost"] = llmcost[new_reactions.index(rxn)]
            # rxn.metadata["llm_response"] = raw_responses[new_reactions.index(rxn)]
            new_nodes = node.add_stub(cost, rxn)
            self._mol_nodes.extend(new_nodes)

    def _filter_reaction(self, reaction: RetroReaction) -> bool:
        if not self.config.filter_policy.selection:
            return False
        try:
            self.config.filter_policy(reaction)
        except RejectionException as err:
            self._logger.debug(str(err))
            return True
        return False

    def _select(self) -> Optional[MoleculeNode]:
        scores = np.asarray(
            [
                node.target_value if node.expandable else np.inf
                for node in self._mol_nodes
            ]
        )

        if scores.min() == np.inf:
            return None

        return self._mol_nodes[int(np.argmin(scores))]

    @staticmethod
    def _update(node: MoleculeNode) -> None:
        v_delta = node.close()
        if node.parent and np.isfinite(v_delta):
            node.parent.update(v_delta, from_mol=node.mol)
