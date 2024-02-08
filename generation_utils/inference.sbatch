#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --error=run-logs/inference-%j.err
#SBATCH --output=run-logs/inference-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=2:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

module purge
module load Python/Anaconda_v03.2023

source deactivate
source activate diffusers

echo "path is "
pwd
nvidia-smi
accelerate env
which python
python -V
conda list
nvcc --version

python3 generation_utils/inference.py \
    --checkpoint $4 \
    --prompts datasets/dreambooth/$3_prompts.json \
    --class_name $2 \
    --descriptor $1 \
    --output_dir $5 \
    --seed 42