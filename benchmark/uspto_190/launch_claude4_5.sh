synthelite_llm_cli \
    --smiles benchmark/uspto_190/targets.csv \
    --save_dir benchmark/uspto_190 \
    --config synthelite_config/configs/synthelite.claude4_5.yml \
    --nproc 2 \
    -skip_tree_if_exist
