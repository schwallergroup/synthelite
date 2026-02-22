from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import List, Set, Tuple

from synthelite.chem.mol import TreeMolecule
from synthelite.chem.reaction import (
    AbsoluteRetroReaction,
    TemplatedRetroReaction,
    TemplatedRetroReactionForMolSet,
)
from synthelite.utils.type_utils import RdMol

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synthelite.search.reaction_mcts.mcts_node import MctsNode

from rdkit import Chem


@dataclass
class BondEdit:
    bonds_lost: Set[Chem.Bond]
    bonds_gained: Set[Chem.Bond]

    def _is_same_added_bond(
        self, bond_added_maps1: list[tuple], bond_added_maps2: list[tuple]
    ):
        """For added bonds, only having one similar index is enough because the atom mapping of the added atoms is not certain."""
        # for map1 in bond_added_maps1:
        #     for map2 in bond_added_maps2:
        #         if map1[0] in map2 or map1[1] in map2:
        #             return True
        # return False
        return min(bond_added_maps1)[0] == min(bond_added_maps2)[0]

    def _is_same_lost_bond(
        self, bond_lost_maps1: list[tuple], bond_lost_maps2: list[tuple]
    ):
        """For lost bonds, both indices of the atoms of a bond must be matched."""
        for map1 in bond_lost_maps1:
            for map2 in bond_lost_maps2:
                if map1 == map2:
                    return True
        return False

    def __eq__(self, other):
        bond_lost_maps = set(_get_bond_atom_maps(bond) for bond in self.bonds_lost)
        other_bond_lost_maps = set(
            _get_bond_atom_maps(bond) for bond in other.bonds_lost
        )
        if bond_lost_maps or other_bond_lost_maps:  # If there is an reaction broken
            return self._is_same_lost_bond(bond_lost_maps, other_bond_lost_maps)
        # else: # no broken bond, which means something was added
        bond_gained_maps = list(_get_bond_atom_maps(bond) for bond in self.bonds_gained)
        other_bond_gained_maps = list(
            _get_bond_atom_maps(bond) for bond in other.bonds_gained
        )
        if bond_gained_maps and other_bond_gained_maps:
            return self._is_same_added_bond(
                bond_gained_maps, other_bond_gained_maps
            )  # The smallest atom map is the one in the product, so we can compare them
        return False

    def __str__(self):
        return f"Bonds lost: {self.bonds_lost_maps},\nBonds gained: {self.bonds_gained_maps}"

    @property
    def bonds_lost_maps(self) -> list[tuple]:
        return list(_get_bond_atom_maps(bond) for bond in self.bonds_lost)

    @property
    def bonds_gained_maps(self) -> list[tuple]:
        return list(_get_bond_atom_maps(bond) for bond in self.bonds_gained)


def _get_bond_atom_maps(bond: Chem.Bond):
    return tuple(
        sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))
    )


def _get_bond_atom_idx(bond: Chem.Bond):
    return tuple(sorted((bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())))


def combine_mols(mols: list[RdMol]):
    res = mols[0]
    if len(mols) == 1:
        return res
    for mol in mols[1:]:
        res = Chem.CombineMols(res, mol)
    return res


def _get_changed_bonds(product_mol: RdMol, reactant_mol: RdMol) -> BondEdit:
    bonds_lost = set()
    bonds_gained = set()
    reactant_mol_atom_idx = set(atom.GetIdx() for atom in reactant_mol.GetAtoms())

    product_atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for _, atom in enumerate(product_mol.GetAtoms())
    }
    reactant_atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for _, atom in enumerate(reactant_mol.GetAtoms())
    }

    def _atom_maps_to_atom_idx(atom_map1: int, atom_map2: int, map_to_id: dict):
        return map_to_id.get(atom_map1, -1), map_to_id.get(atom_map2, -1)

    for bond in product_mol.GetBonds():
        bond_mapped_nums = _get_bond_atom_maps(bond)
        # bond_idx = (product_atom_map_to_idx[bond_mapped_nums[0]], product_atom_map_to_idx[bond_mapped_nums[1]])
        bond_idx = _atom_maps_to_atom_idx(*bond_mapped_nums, reactant_atom_map_to_idx)
        if (
            bond_idx[0] not in reactant_mol_atom_idx
            or bond_idx[1] not in reactant_mol_atom_idx
        ):
            continue
        corresponding_bond = reactant_mol.GetBondBetweenAtoms(*bond_idx)
        if (
            corresponding_bond
            and bond.GetBondType() != corresponding_bond.GetBondType()
        ):
            bonds_lost.add(bond)
            bonds_gained.add(corresponding_bond)

    return bonds_lost, bonds_gained


def _get_bond_edits(
    products: list[TreeMolecule], reactants: list[TreeMolecule]
) -> BondEdit:
    product_mol = combine_mols([each.mapped_mol for each in products])
    reactant_mol = combine_mols([each.mapped_mol for each in reactants])

    product_atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for _, atom in enumerate(product_mol.GetAtoms())
    }
    reactant_atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for _, atom in enumerate(reactant_mol.GetAtoms())
    }

    prod_bond_maps = set()
    for bond in product_mol.GetBonds():
        bond_map_nums = _get_bond_atom_maps(bond)
        prod_bond_maps.add(bond_map_nums)
    reactant_bond_maps = set()
    for bond in reactant_mol.GetBonds():
        bond_map_nums = _get_bond_atom_maps(bond)
        reactant_bond_maps.add(bond_map_nums)

    bond_lost_maps = prod_bond_maps - reactant_bond_maps
    bond_gained_maps = reactant_bond_maps - prod_bond_maps

    bond_lost_idx = [
        (product_atom_map_to_idx[map_num1], product_atom_map_to_idx[map_num2])
        for map_num1, map_num2 in bond_lost_maps
    ]
    bond_gained_idx = [
        (reactant_atom_map_to_idx[map_num1], reactant_atom_map_to_idx[map_num2])
        for map_num1, map_num2 in bond_gained_maps
    ]

    # return bond_lost_maps, bond_gained_maps

    bonds_lost = set([product_mol.GetBondBetweenAtoms(*idx) for idx in bond_lost_idx])
    bonds_gained = set(
        [reactant_mol.GetBondBetweenAtoms(*idx) for idx in bond_gained_idx]
    )

    bonds_lost_by_type, bonds_gained_by_type = _get_changed_bonds(
        product_mol, reactant_mol
    )

    bonds_lost = bonds_lost.union(bonds_lost_by_type)
    bonds_gained = bonds_gained.union(bonds_gained_by_type)

    return BondEdit(bonds_lost=bonds_lost, bonds_gained=bonds_gained)


# def _get_retro_action_changed_bonds(action: AbsoluteRetroReaction):
#     products = [action.mol]
#     reactants = action.reactants
#     return _get_changed_bonds(products, reactants)

# Global cache for bond changes
_bond_changes_cache = {}


def _get_action_hash(action) -> str:
    """Create a hash key for an action to use in caching."""
    try:
        # Use molecule SMILES and reactant SMILES for hashing
        if not isinstance(action, TemplatedRetroReactionForMolSet):
            product_smiles = action.mol.smiles
            reactant_smiles = [
                ".".join(mol.smiles for mol in reactants)
                for reactants in action.reactants
            ]
        else:
            product_smiles = ".".join(mol.smiles for mol in action.products)
            reactant_smiles = [
                ".".join(mol.smiles for mol in reactants)
                for reactants in action.reactants
            ]

        # Create a deterministic hash
        hash_input = f"{product_smiles}|{sorted(reactant_smiles)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception:
        # Fallback to object id if hashing fails
        return str(id(action))


def get_action_changed_bond(action) -> list[BondEdit]:
    # Try to get from cache first
    action_hash = _get_action_hash(action)
    if action_hash in _bond_changes_cache:
        return _bond_changes_cache[action_hash]

    # Compute bond changes
    if not isinstance(action, TemplatedRetroReactionForMolSet):
        product_mol = action.mol
        reactants_sets = action.reactants
        bond_changes = [
            _get_bond_edits([product_mol], reactants_set)
            for reactants_set in reactants_sets
        ]
    else:
        reactants_sets = action.reactants
        bond_changes = [
            _get_bond_edits([product], reactants_set)
            for product, reactants_set in zip(action.products, action.reactants)
        ]

    # Cache the result
    _bond_changes_cache[action_hash] = bond_changes
    return bond_changes


def clear_bond_changes_cache():
    """Clear the bond changes cache."""
    global _bond_changes_cache
    _bond_changes_cache.clear()


def get_bond_changes_cache_stats():
    """Get statistics about the bond changes cache."""
    return {"cache_size": len(_bond_changes_cache)}


def is_equal_disconnection(action1, action2):
    bond_changes1 = get_action_changed_bond(action1)
    bond_changes2 = get_action_changed_bond(action2)
    for bond_change1 in bond_changes1:
        for bond_change2 in bond_changes2:
            if bond_change1 == bond_change2:
                return True
    return False


def get_actual_change(
    action: TemplatedRetroReaction, node: "MctsNode"
) -> Tuple[BondEdit, List[TreeMolecule]]:
    """Get the actual reactants for an action by comparing with the child node's state."""
    bond_changes = get_action_changed_bond(action)
    action_reactants = action.reactants

    parent = node.parent

    actual_bond_change = _get_bond_edits(
        products=parent.state.mols, reactants=node.state.mols
    )

    for bond_change, action_reactant in zip(bond_changes, action_reactants):
        if bond_change == actual_bond_change:
            return bond_change, action_reactant

    return None, None


def apply_sequence_of_actions(
    target_node: "MctsNode",
    actions: List[TemplatedRetroReaction],
    node_path: List["MctsNode"],
):
    """Attempt to apply a sequence of actions to a node, and return actions that cannot be applied or applied in a different way than the original disconnection.

    node_path is required since one action can yield multiple outcomes.
    """
    res = []
    current_mol = target_node.state.mols
    for action, node in zip(actions, node_path):
        smart = action.smarts
        new_action = TemplatedRetroReactionForMolSet(
            mol_set=current_mol,
            smarts=smart,
            metadata={},
            use_rdchiral=True,
        )
        # print("=====")
        # for product, reactants in zip(new_action.products, new_action.reactants):
        #     print(f"{product.smiles}>>{'.'.join([reactant.smiles for reactant in reactants])}", is_equal_disconnection(action, new_action))

        if not is_equal_disconnection(action, new_action):
            res.append(action)
            continue

        # bond_changes, action_reactants = get_actual_change(action)
        actual_bond_change, actual_reactants = get_actual_change(action, node)
        new_bond_changes = get_action_changed_bond(new_action)
        new_action_reactant_sets = new_action.reactants

        for new_bond_change, new_action_reactants, new_product in zip(
            new_bond_changes, new_action_reactant_sets, new_action.products
        ):
            if new_bond_change == actual_bond_change:
                actual_new_reactants = new_action_reactants
                actual_product = new_product
                break

        keep_mols = [mol for mol in current_mol if mol is not actual_product]

        current_mol = keep_mols + list(actual_new_reactants)

    return res


def map_reaction_dependance(node: "MctsNode"):
    res = {}
    actions, node_path = node.path_to()
    root = node_path.pop(0)

    for i, action in enumerate(actions):
        remaining_actions = actions[:i] + actions[i + 1 :]
        remaining_nodes = node_path[:i] + node_path[i + 1 :]

        affected_actions = apply_sequence_of_actions(
            target_node=root, actions=remaining_actions, node_path=remaining_nodes
        )
        if affected_actions:
            res[action] = affected_actions
        else:
            res[action] = []

    return res
