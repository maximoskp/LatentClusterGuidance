#!/bin/bash

# TODO:
# 1) Correct the transformer model so that it isolates the FiLM
# parameters in a ModuleList.
# 2) Add an argument for selecting contrastive vs MSE loss.

# List of Python scripts with their respective arguments
scripts=(
    # "generate_nott2jazz.py -l 0 -s 0.5 -g 0"
    # "generate_nott2jazz.py -l 0 -s 1.0 -g 0"
    
    # "generate_nott2jazz.py -l 4 -s 0.5 -g 0"
    # "generate_nott2jazz.py -l 4 -s 1.0 -g 0"

    # "generate_nott2jazz.py -l 6 -s 0.5 -g 0"
    # "generate_nott2jazz.py -l 6 -s 1.0 -g 0"

    # "generate_nott2jazz.py -l -1 -s 0.5 -g 0"
    # "generate_nott2jazz.py -l -1 -s 1.0 -g 0"

    "generate_jazz2nott.py -l 0 -s 0.5 -g 2"
    "generate_jazz2nott.py -l 0 -s 1.0 -g 2"
    
    "generate_jazz2nott.py -l 4 -s 0.5 -g 2"
    "generate_jazz2nott.py -l 4 -s 1.0 -g 2"

    "generate_jazz2nott.py -l 6 -s 0.5 -g 2"
    "generate_jazz2nott.py -l 6 -s 1.0 -g 2"

    "generate_jazz2nott.py -l -1 -s 0.5 -g 2"
    "generate_jazz2nott.py -l -1 -s 1.0 -g 2"
)

# Name of the conda environment
conda_env="torch"

# Path to global conda
CONDA_SH="/opt/miniconda3/etc/profile.d/conda.sh"

for script in "${scripts[@]}"; do
    script_name=$(echo "$script" | awk '{print $1}')
    screen_name=$(basename "$script_name" .py)

    screen -dmS "$screen_name" bash -c "
        source \"$CONDA_SH\"
        conda activate \"$conda_env\"
        python $script
        exec bash
    "

    echo "Started screen '$screen_name' for script '$script'."
done