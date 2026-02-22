import argparse
import asyncio
import os
from typing import Optional

from synthelite.chem.mol import Molecule
from synthelite.synthelite_llm_finder import SyntheliteLLMFinder

import pandas as pd

from synthelite.utils.clogging import init_logger

from dotenv import load_dotenv
from synthelite.utils.files import split_file, start_processes

load_dotenv()

# global_log_file = 'synthelite.global.log'


def parse_args():
    parser = argparse.ArgumentParser(
        "synthelite_llm_cli", description="Run beam search with LLM guidance."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="Root SMILES for the search tree, or a .csv containing columns idx, smiles, steer_query.",
    )
    parser.add_argument("--steer_query", type=str, help="Query to guide the search.")
    parser.add_argument("--save_dir", type=str, help="Path to save the results.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
    )
    # parser.add_argument('--resume_llm_guided_tree', type=str, default=None,
    # help='Path to a previously saved tree to resume from. Tree must not enter MCTS phase.')
    parser.add_argument(
        "--error_log_file",
        type=str,
        default="errors.csv",
    )

    # parser.add_argument(
    #     '--prune_beam_search_steps',
    #     type=int,
    #     nargs='*',
    #     help='List of maximum beam search steps to prune the tree before MCTS. Default is None.')

    parser.add_argument(
        "-skip_tree_if_exist",
        action="store_true",
        help="Skip the LLM-guided beam search phase. Only viable if --resume_llm_guided_tree is provided.",
    )

    parser.add_argument(
        "--nproc",
        type=int,
        help="if given, the input is split over a number of processes",
    )
    # parser.add_argument('--prune_beam_search_steps', type=int, default=[10], nargs='*', help='List of maximum beam search steps to prune the tree before MCTS. Default is [10].')
    # parser.add_argument('--llm_guided_suffix', type=str, default='llm_guided.json')
    # parser.add_argument('--mcts_suffix', type=str, default='mcts.json')
    return parser.parse_args()


async def _process_single_smiles(
    smiles: str,
    finder: SyntheliteLLMFinder,
    save_dir: str,
    steer_query: Optional[str] = None,
    # checkpoint_tree: Optional[str] = None,
    skip_tree_if_exist: bool = True,
    # output_name: str,
    # do_clustering: bool,
    # route_distance_model: Optional[str],
    # post_processing: List[_PostProcessingJob],
    # pre_processing: Optional[_PreProcessingJob],
):
    # logger = init_logger(os.path.join(save_dir, 'synthelite.log'))

    finder.target_smiles = smiles
    finder.steer_query = steer_query
    finder.save_dir = save_dir

    # finder.prepare_tree(checkpoint_tree)

    await finder.tree_search(
        skip_tree_if_exist=skip_tree_if_exist,
    )


async def _process_multi_smiles(
    filename: str,
    finder: SyntheliteLLMFinder,
    save_dir: str,
    skip_tree_if_exist: bool = True,
    error_log_file: str = "errors.csv",
):
    logger = init_logger()
    # try:
    input_smiles = pd.read_csv(filename)
    error_data = []
    for row in input_smiles.itertuples():
        try:
            idx, smiles, steer_query = row.idx, row.smiles, row.steer_query
            steer_query = steer_query if pd.notna(steer_query) else None
            this_save_dir = os.path.join(save_dir, f"{idx}")
            finder.save_dir = this_save_dir
            # checkpoint_tree_path = finder.tree_llm_guided_path if skip_tree_if_exist else None
            await _process_single_smiles(
                smiles=smiles,
                finder=finder,
                save_dir=this_save_dir,
                steer_query=steer_query,
                # checkpoint_tree=checkpoint_tree_path,
                skip_tree_if_exist=skip_tree_if_exist,
            )
        except Exception as e:
            logger.error(
                f"Processing SMILES {smiles} (idx={idx}) failed with exception: {e}",
                exc_info=True,
            )
            error_data.append((idx, smiles, steer_query, str(e)))
            continue
    if error_data:
        error_df = pd.DataFrame(
            error_data, columns=["idx", "smiles", "steer_query", "error"]
        )
        error_df.to_csv(os.path.join(save_dir, error_log_file), index=False)
    # except Exception as e:
    #     logger.error(f"Processing multiple SMILES failed with exception: {e}", exc_info=True)
    #     raise e


def _multiprocess_smiles(args: argparse.Namespace) -> None:
    def create_cmd(index, filename):
        cmd_args = [
            "synthelite_llm_cli",
            "--smiles",
            filename,
            "--config",
            args.config,
            "--save_dir",
            args.save_dir,
            "--error_log_file",
            f"errors.{index}.csv",
        ]
        if args.skip_tree_if_exist:
            cmd_args.append("-skip_tree_if_exist")
        return cmd_args

    if not os.path.exists(args.smiles):
        raise ValueError(
            "For multiprocessing execution the --smiles argument needs to be a filename"
        )

    filenames = split_file(args.smiles, args.nproc)
    log_prefix = os.path.join(args.save_dir, "synthelite_llm_cli")
    start_processes(filenames, log_prefix, create_cmd)


def main():
    args = parse_args()
    logger = init_logger()

    if not os.path.exists(args.smiles):
        mol = Molecule(smiles=args.smiles)
        if mol.rd_mol is None:
            logger.error(
                f"The --smiles argument ({args.smiles})"
                " does not point to an existing file or is a valid RDKit SMILES."
                " Cannot start retrosynthesis planning."
            )
            return

    if args.nproc:
        _multiprocess_smiles(args)
        return

    finder = SyntheliteLLMFinder(configfile=args.config)

    multi_smiles = os.path.exists(args.smiles)

    if multi_smiles:
        coron = _process_multi_smiles(
            filename=args.smiles,
            finder=finder,
            save_dir=args.save_dir,
            skip_tree_if_exist=args.skip_tree_if_exist,
        )
    else:
        coron = _process_single_smiles(
            smiles=args.smiles,
            finder=finder,
            save_dir=args.save_dir,
            steer_query=args.steer_query,
            # checkpoint_tree=args.resume_llm_guided_tree,
            skip_tree_if_exist=args.skip_tree_if_exist,
        )

    asyncio.run(coron)


if __name__ == "__main__":
    main()
