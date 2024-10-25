#!/bin/sh

# Bash script to run multiple Python codes.

# ########### ACTIVATE ENV #############################
# Source the conda setup script to enable 'conda activate' in a non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Now activate your environment
conda activate qbio_paper_env
export CUDA_VISIBLE_DEVICES=0,1

# Ensure PYTHONPATH includes your project directory
PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT_DIR}/src"

# Kill any existing instances of the script to prevent duplicates
pkill -f "parameter_estimation_constitutive.py"

# Remove previous output
rm -f output.txt

# ########### PROGRAM ARGUMENTS #############################
# Paths with configuration files
path_to_executable="${PROJECT_ROOT_DIR}/notebooks/parameter_estimation_constitutive.py"
chain_length=1100000

# Run the Python script with nohup using 'python' from the activated environment
nohup python "$path_to_executable" $chain_length > output.txt 2>&1 &

# Deactivating the environment after starting the process
conda deactivate

exit 0


# ########### TO EXECUTE RUN IN TERMINAL #########################
# run as: source runner.sh /dev/null 2>&1 & disown

# ########### TO MONITOR PROGRESS #########################
# To check if the process is still running
# ps -ef | grep python3