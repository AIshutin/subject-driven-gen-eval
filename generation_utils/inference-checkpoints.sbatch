#!/bin/bash
#SBATCH --job-name=inference_checkpoints
#SBATCH --error=run-logs/inference_checkpoints-%j.err
#SBATCH --output=run-logs/inference_checkpoints-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=8:00:00
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

bash generation_utils/inference-checkpoints.sh $1 $2 $3 $4 $5