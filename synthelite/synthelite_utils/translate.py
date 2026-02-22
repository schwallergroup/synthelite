"""Translate ReactionTree format into Synthesis Challenge."""

import json
import pandas as pd
import networkx as nx
from synthelite.chem import UniqueMolecule
from synthelite.reactiontree import ReactionTree
from pydantic import BaseModel
from typing import List


class SynthesisChallenge(BaseModel):
    standard_conditions: List[dict]

    def convert(self, tree):
        """Convert ReactionTree to Synthesis Challenge format."""
        graph = tree.graph
        new_graph = self._drop_unique_molecule(graph)
        dependencies = self._get_dependencies(new_graph)
        steps = self._get_steps(new_graph)

        return {"dependencies": dependencies, "steps": steps}

    def _drop_unique_molecule(self, graph):
        """Make graph without FixedRetroReaction nodes."""

        G = nx.DiGraph()
        for node in graph.nodes:
            if isinstance(node, UniqueMolecule):
                innodes = list(graph.predecessors(node))
                outnodes = list(graph.successors(node))
                if innodes:
                    for outnode in outnodes:
                        for innode in innodes:
                            G.add_edge(innode, outnode)

        # Populate G with more information
        for i, node in enumerate(G.nodes):
            # Add ids and molecules from `graph`
            product = list(graph.predecessors(node))
            reactants = list(graph.successors(node))

            nx.set_node_attributes(
                G,
                {
                    node: {
                        "product": product,
                        "reactants": reactants,
                        "reaction": node,
                        "metadata": graph.nodes[product[0]],
                        "id": str(i + 1),
                    }
                },
            )
        return G

    def _get_steps(self, graph):
        steps = []
        for node, data in graph.nodes(data=True):

            # TODO: reactionmetrics turn into callable
            step = {
                "step": data["id"],
                "reactants": self._build_mols(data["reactants"], name="reactant"),
                "reagents": data["metadata"]["metadata"].get("conditions", {}),
                "products": self._build_mols(data["product"], name="product"),
                "conditions": self.standard_conditions,
                "reactionmetrics": data["metadata"]["metadata"].get(
                    "reactionmetrics", {}
                ),
                "__metadata": self._get_metadata(data),
            }
            steps.append(step)
        return steps

    def _get_dependencies(self, graph):
        """Compute dependency matrix"""
        dependencies = {
            data["id"]: [
                graph.nodes[successor]["id"] for successor in graph.successors(node)
            ]
            for node, data in graph.nodes(data=True)
        }
        return dependencies

    def _build_mols(self, mols, name="reactant"):
        return [{"smiles": mol.smiles, f"{name}_metadata": {}} for mol in mols]

    def _get_metadata(self, data) -> dict:
        return data["reaction"].metadata


if __name__ == "__main__":

    data = pd.read_json("mol1_output.json", orient="table")
    all_trees = (
        data.trees.values
    )  # This contains a list of all the trees for all the compounds
    trees_for_first_target = all_trees[0]

    converter = SynthesisChallenge(
        standard_reagents=[{"smiles": "sssss", "reagent_metadata": {}}],
        standard_conditions=[{"temperature": 25, "pressure": 1}],
        standard_metrics=[
            {
                "scalabilityindex": "5",
                "confidenceestimate": "0.75",
                "closestliterature": "DOI:###",
            }
        ],
    )

    for itree, tree in enumerate(trees_for_first_target):
        reaction_tree = ReactionTree.from_dict(tree)
        out = converter.convert(reaction_tree)

        print(json.dumps(out, indent=2))
