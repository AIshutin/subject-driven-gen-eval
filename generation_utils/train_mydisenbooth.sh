export WANDB_MODE="offline"
export SUBJECT_NAME=$1
export CONCEPT_NAME="$2</w>" # like in original code.
export CLASS_NAME=$2
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="datasets/dreambooth/${SUBJECT_NAME}"
export OUTPUT_DIR="checkpoints/mydisenbooth/${SUBJECT_NAME}/sd2.1"
export WANDB_NAME="mydisenbooth-sd2.1-${SUBJECT_NAME}-${CLASS_NAME}"

accelerate launch generation_utils/train_mydisenbooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
  --resolution=512  \
  --gradient_accumulation_steps=1 \
  --train_batch_size=1  \
  --learning_rate=1e-4 \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --report_to wandb \
  --sample_batch_size=10 \
  --seed=42 \
  --validation_prompt "a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
  --checkpointing_steps 200 \
  --validation_steps 200 \
  --num_validation_images 3 \
  --validation_prompt "a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
                      "a ${CONCEPT_NAME} ${CLASS_NAME//_/ } at school classroom" \
                      "a bronze statue of ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
                      "a ${CONCEPT_NAME} ${CLASS_NAME//_/ } with the Tower of Pisa in the background" \
                      "a green ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \