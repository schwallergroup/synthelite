# Synthelite: Chemist-aligned and feasibility-aware synthesis planning with LLMs

[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2512.16424.svg)](https://doi.org/10.48550/arXiv.2512.16424)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)



## Overview
Synthelite is a Computer-Aided Synthesis Planning (CASP) software central around LLMs. While LLMs are potent reaction policy, their high computational cost hinder their use in traditional CASP tools which are typically based on explorative search with cheap policies.
Synthelite overcomes this issues by separating the LLMs from the search: LLMs act as a master planner, decide which bonds to cut and what kind of reactions should be done at each step, and a second phase using Monte-Carlo Tree Search (MCTS) to search for a sequence of reactions that match the strategy of the LLMs and lead the search to in-stock materials.

The cool thing about using LLMs as synthesis planner is that it enables a seamless interaction interface with users.
Besides the target molecule, Synthelite allows additional constrains from chemists under a short natural language prompt.
Moreover, the chemistry knowledge of the LLMs allows them plan the synthesis with intention and chemical-feasibility awareness, in constrast to the randomness of traditional CASP tools.

For more details, checkout [our preprint](https://arxiv.org/abs/2512.16424).


## Installation

First clone the repository using Git, then execute the following commands in the root of the repository

    conda env create -f env-dev.yml
    conda activate synthelite
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    poetry install --all-extras

the `synthelite` package is now installed in editable mode.

To use the tool you need:

1. An `.env` file containing an OpenRouter key as `OPENROUTER_API_KEY`, an OpenAI key (for query embedding) stored as `OPENAI_API_KEY`, and the location on WandB where you want to stream out the LLMs output as `WANDB_PROJECT`.
2. A stock file containing buyable molecules. We use `eMolecule` for our experiments.
3. An LLM-annotated reaction templates.
4. AiZynthFinder policies model. The policies are used as fallback to propose reactions in cases the template search process fails to find one that matches the LLMs' strategy. The final reaction is still selected by the LLMs.

To set up `.env` and download the necessary files, run:
```bash
sh set_up.sh
```

Or if you already have the `.env` file and only wish to download the others:
```
download_public_data data/
```
If you want to install the files elsewhere rather than `data/`, change the file locations accordingly in the config files (see [synthelite_config/configs](synthelite_config/configs)).

Also make sure you have logged in to your WandB account to track the LLM traces.

## Usage

Synthelite is runnable with a CLI, requiring a config file and a input file in `.csv` format, containing one or multiple pairs of target-prompt.
The `.csv` file must contains the following columns:

- `idx`: index of the target-prompt pair, used as the directory name containing this case results.
- `smiles`: SMILES string of the target.
- `steer_query`: Synthesis constrains in natural language format.

A typical launch would look like:
```bash
synthelite_llm_cli \
    --smiles example/simple_launch/targets.csv \
    --save_dir example/simple_launch \
    --config synthelite_config/configs/synthelite.claude4_5.yml \
    -skip_tree_if_exist
```
A result directory will be created for each case, containing multiple `.json` files storing the information of the search tree and routes.
The final routes are stored in file `routes.llm_query_explorer.json`.

For more information, please take a look at the example in `example/simple_launch`.

## Experiments

Full reproduction of the results in the preprint requires considerable time and API budget.
We therefore provide the precomputed routes used to produce the figures in the paper on HuggingFace at [`SchwallerGroup/synthelite`](https://huggingface.co/datasets/SchwallerGroup/synthelite).

The routes are organized by experiment and model:
```
routes/
├── strategic/          # Strategic synthesis planning experiments
│   ├── gemini2_5/
│   ├── claude4_5/
│   └── gpt5/
├── starting_materials/ # Starting-material-constrained experiments
│   ├── gemini2_5/
│   ├── claude4_5/
│   └── gpt5/
└── uspto_190/          # USPTO-190 benchmark (zipped)
    ├── gemini2_5.zip
    ├── claude4_5.zip
    └── gpt5.zip
```

To download all routes:
```bash
download_routes --output_dir data/routes
```

You can also download a subset by specifying experiments and/or models:
```bash
# Only the strategic experiment with Claude 4.5
download_routes --output_dir data/routes --experiments strategic --models claude4_5

# USPTO-190 results for all models
download_routes --output_dir data/routes --experiments uspto_190
```

Zip files are automatically extracted after download. Use `--no-unzip` to skip extraction.

## Acknowledgement

Synthelite codebase is a heavily-modified fork of [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder) [2] by MolecularAI.
We appreciate the authors for the clean implementation of AiZynthFinder.

## License

The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.

## References

1. Bran, Andres M., et al. "Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation." arXiv preprint arXiv:2503.08537 (2025).
2. Genheden, Samuel, et al. "AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning." Journal of cheminformatics 12.1 (2020): 70.

## Citations
```txt
@article{xuan2025synthelite,
  title={Synthelite: Chemist-aligned and feasibility-aware synthesis planning with LLMs},
  author={Xuan-Vu, Nguyen and Armstrong, Daniel and Wehrbach, Milena and Bran, Andres M and Jon{\v{c}}ev, Zlatko and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2512.16424},
  year={2025}
}
```
