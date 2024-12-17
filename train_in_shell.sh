#!/bin/bash

# Activate the Conda environment
source /fs/classhomes/fall2024/cmsc848k/c848k032/miniconda3/bin/activate obitonet

# Navigate to the project directory
cd /fs/classhomes/fall2024/cmsc848k/c848k032/ObitoNet

# Load required modules
module add cuda/11.8.0
module add gcc/11.2.0

# Run the main script with the specified configuration
python main.py --launcher pytorch --config configs/config.yaml --exp_name CA_Train_1_DISTRIBUTED

# python visualization.py --test --config configs/config.yaml --exp_name CA_Train_1_BASELINE
# Print success message
echo "ran successfully"
