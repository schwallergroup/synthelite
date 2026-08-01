# Import sqlite3 FIRST, before pandas. pandas loads the system libstdc++, after
# which sqlite3 (pulled in later via synthelite -> weave -> diskcache) fails to
# find CXXABI_1.3.15 for the conda libicu. Loading it first avoids the clash.
import sqlite3  # noqa: F401  (kept first for load-order; not used directly)

import os
from typing import Optional
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import argparse

from synthelite.reactiontree import ReactionTree
from functions.synthelite_check import apply_func_to_route

# ROUTE_DIR = "/data/vu/synthegy/results/query_explore_with_exact_site"
# directory contains path to route in the format like `steer_1.json`, `synthegy_1.json`,...

check_functions_path = "evaluation_cases.csv"
# out_path = os.path.join(ROUTE_DIR, "benchmark_azf_results.csv")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route_dir", type=str, required=True)
    parser.add_argument("--outfile", type=str, default="benchmark_results.csv")
    parser.add_argument("--check_functions_path", type=str, default=check_functions_path)
    parser.add_argument("--allow_neural", action="store_true")
    return parser.parse_args()

def get_routes(route_dir: str, case_id: str, limits: Optional[int] = 50, allow_neural: bool=True):
    def _has_neural_policy(route):
        route_obj = ReactionTree.from_dict(route)
        for action in route_obj.reactions():
            if action.metadata.get("policy_name") != "llm_query_explorer":
                return True
        return False
    
    route_path = os.path.join(route_dir, f"{case_id}/routes.llm_query_explorer.json")
    with open(route_path, 'r') as f:
        routes = json.load(f)
    
    if not allow_neural:
        routes = [r for r in routes if not _has_neural_policy(r)]
    
    if limits is not None:
        routes = routes[:limits]
    return routes

def main():
    args = parse_args()
    out_path = os.path.join(args.route_dir, args.outfile)
    check_function_df = pd.read_csv(check_functions_path)
    
    if os.path.exists(out_path):
        processed_results_df = pd.read_csv(out_path)
        processed_results_df = processed_results_df.set_index("case_id")
    else:
        processed_results_df = None    
    
    results = []
    
    for row in check_function_df.itertuples():
        func_name = row.func_name
        if pd.isna(func_name):
            continue
        case_id = row.case_id.strip("'")
        
        if processed_results_df is not None and case_id in processed_results_df.index:
            processed_row = processed_results_df.loc[case_id]
            results.append(
                {
                    "case_id": case_id,
                    "func_name": processed_row.func_name,
                    "route_scores": processed_row.route_scores,
                    "success_rate": processed_row.success_rate,
                    "n_routes": len(json.loads(processed_row.route_scores))
                }
            )
            continue
        
        
        print("Calculating scores for case:", case_id, "with function:", func_name)
        
        try:
            routes = get_routes(args.route_dir, case_id, limits=50, allow_neural=args.allow_neural)
            if len(routes) == 0:
                print(f"No routes found for case {case_id} with function {func_name}")
                continue
            
            route_scores = [apply_func_to_route(func_name, route) for route in tqdm(routes)]
            route_scores = list(map(int, route_scores))
            success_rate = sum(route_scores) / len(route_scores) if route_scores else 0
            
            results.append(
                {
                    "case_id": case_id,
                    "func_name": func_name,
                    "route_scores": route_scores,
                    "success_rate": success_rate,
                    "n_routes": len(route_scores)
                }
            )
        except Exception as e:
            print(f"Error processing case {case_id} with function {func_name}: {e}")
            continue
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    
if __name__ == "__main__":
    main()