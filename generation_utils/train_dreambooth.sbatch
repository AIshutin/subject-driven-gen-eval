#!/bin/bash
#SBATCH --job-name=dreambooth-berry_bowl
#SBATCH --error=run-logs/dreambooth-berry_bowl-%j.err
#SBATCH --output=run-logs/dreambooth-berry_bowl-%j.log
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

export PROMPTS=datasets/dreambooth/creature_prompts.json
export CLASS_NAME=bowl
export SUBJECT_NAME=berry_bowl
export OUTPUT_DIR="checkpoints/${SUBJECT_NAME}/db-sd2.1"

bash generation_utils/train_dreambooth.sh $SUBJECT_NAME $CLASS_NAME

declare -a arr=("step800" "step1000" "step1200" "step1400" "step1600")
for i in "${arr[@]}"
do
   python3 generation_utils/inference.py --checkpoint "$OUTPUT_DIR/$i" \
                        --prompts $PROMPTS \
                        --class_name $CLASS_NAME \
                        --descriptor htazawa \
                        --output_dir generated/$SUBJECT_NAME/dreambooth-sd2.1-$i
done

rm -r $OUTPUT_DIR
