synthelite_llm_cli \
    --smiles data/benchmark/strategic/targets.csv \
    --save_dir data/benchmark/strategic \
    --config synthelite_config/configs/synthelite.claude4_5.yml \
    --nproc 2 \
    -skip_tree_if_exist
