synthelite_llm_cli \
    --smiles benchmark/starting_materials/targets.csv \
    --save_dir benchmark/starting_materials \
    --config synthelite_config/configs/synthelite.claude4_5.yml \
    --nproc 2 \
    -skip_tree_if_exist
