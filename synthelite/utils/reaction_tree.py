import json
from synthelite.reactiontree import ReactionTree


def deduplicate_routes(routes: list[ReactionTree]) -> list[ReactionTree]:
    seen = set()
    unique_routes = []
    for route in routes:
        if route.hash_key() in seen:
            continue
        seen.add(route.hash_key())
        unique_routes.append(route)

    return unique_routes


def load_routes(route_json_file: str, to_dict: bool = True):
    with open(route_json_file, "r") as file:
        routes = json.load(file)
    routes = [ReactionTree.from_dict(route) for route in routes]
    routes = deduplicate_routes(routes)
    if to_dict:
        return [each.to_dict() for each in routes]
    else:
        return routes
