#!/bin/bash
#SBATCH --job-name=dreambooth
#SBATCH --error=run-logs/dreambooth-%j.err
#SBATCH --output=run-logs/dreambooth-%j.log
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

export PROMPTS=datasets/dreambooth/$3_prompts.json
export SUBJECT_NAME=$1
export CLASS_NAME=$2
export OUTPUT_DIR="checkpoints/dreambooth/${SUBJECT_NAME}/sd2.1"

bash generation_utils/train_dreambooth.sh $SUBJECT_NAME $CLASS_NAME

declare -a arr=("800" "1000" "1200" "1400" "1600")
for i in "${arr[@]}"
do
   python3 generation_utils/inference.py --checkpoint "$OUTPUT_DIR/checkpoint-$i" \
                        --prompts $PROMPTS \
                        --class_name $CLASS_NAME \
                        --descriptor htazawa \
                        --output_dir generated/dreambooth/$SUBJECT_NAME/sd2.1-step$i \
                        --seed 42
   rm -r "$OUTPUT_DIR/checkpoint-$i"
done
