import json


def flip_reactions(route):
    route_copy = json.loads(json.dumps(route))  # Deep copy to avoid modifying the original route
    def dfs_traverse(node, depth=0):
        if node["type"] == "reaction":
            try:
                # Extract reactants and product
                rsmi = node["metadata"]["mapped_reaction_smiles"]
                # reactants_part = rsmi.split(">")[0]
                # product_part = rsmi.split(">")[-1]
                products, reagents, reactants = rsmi.split(">")
                node["metadata"]["mapped_reaction_smiles"] = f'{reactants}>{reagents}>{products}'

            except Exception as e:
                print(f"Error processing reaction node: {e}")

        # Determine the next depth based on the current node's type
        next_depth = depth
        if node["type"] != "reaction": # This means it's a chemical node
            next_depth = depth + 1

        # Continue traversing the tree
        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    # Start traversal from the root
    dfs_traverse(route_copy)
    return route_copy

def get_route_len(route):
    max_depth = 0

    def dfs_traverse(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)

        # Determine the next depth based on the current node's type
        next_depth = depth
        if node["type"] != "reaction":  # This means it's a chemical node
            next_depth = depth + 1

        # Continue traversing the tree
        for child in node.get("children", []):
            dfs_traverse(child, next_depth)

    # Start traversal from the root
    dfs_traverse(route)
    return max_depth