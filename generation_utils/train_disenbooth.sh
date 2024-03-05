export WANDB_MODE="offline"
export SUBJECT_NAME=$1
export CONCEPT_NAME="htazawa" # sts is bad one, since it's a rifle
export CLASS_NAME=$2
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="datasets/dreambooth/${SUBJECT_NAME}"
export CLASS_DIR="../synth-dataset-sd-2-1/${CLASS_NAME}"
export OUTPUT_DIR="checkpoints/custom_diffusion/${SUBJECT_NAME}/sd2.1"
export WANDB_NAME="cd-sd2.1-${SUBJECT_NAME}-${CLASS_NAME}"

accelerate launch generation_utils/train_disenbooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a $CONCEPT_NAME ${CLASS_NAME//_/ }" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_epochs=200 \
  --seed="0" \
  --report_to wandb \
  --seed=42 \
  --validation_prompt "a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
                      "a ${CONCEPT_NAME} ${CLASS_NAME//_/ } at school" \
                      "a ${CONCEPT_NAME} ${CLASS_NAME//_/ } in bronze" \
                      "a ${CONCEPT_NAME} ${CLASS_NAME//_/ } with the Tower of Pisa in the background" \
                      "a green ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
  --num_validation_images 3 \
  --mixed_precision no