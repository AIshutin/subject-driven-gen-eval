#!/bin/bash
#SBATCH --job-name=mydisenbooth
#SBATCH --error=run-logs/mydisenbooth-%j.err
#SBATCH --output=run-logs/mydisenbooth-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
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

export SUBJECT_NAME=$1
export CLASS_NAME=$2

bash generation_utils/train_mydisenbooth.sh $SUBJECT_NAME $CLASS_NAME
