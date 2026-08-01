import os
from typing import Optional
import pandas as pd
import json
from glob import glob
from tqdm import tqdm

from functions.synthelite_check import apply_func_to_route

ROUTE_DIR = "/data/vu/synthegy/results/azf/routes"
# directory contains path to route in the format like `steer_1.json`, `synthegy_1.json`,...

check_functions_path = "evaluation_cases.csv"
out_path = os.path.join(ROUTE_DIR, "benchmark_azf_results.csv")

def get_routes(target_id: str, limits: Optional[int] = 50):
    route_path = os.path.join(ROUTE_DIR, f"{target_id}.json")
    with open(route_path, 'r') as f:
        routes = json.load(f)
        
    if limits is not None:
        routes = routes[:limits]
    return routes
    


def main():
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
        
        target_id = case_id[:-1]
        routes = get_routes(target_id, limits=50)
        
        print("Calculating scores for case:", case_id, "with function:", func_name)
        
        try:
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