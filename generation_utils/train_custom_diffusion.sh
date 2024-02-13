export WANDB_MODE="offline"
export SUBJECT_NAME=$1
export CONCEPT_NAME="<htazawa>" # sts is bad one, since it's a rifle
export CLASS_NAME=$2
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="datasets/dreambooth/${SUBJECT_NAME}"
export CLASS_DIR="../synth-dataset-sd-2-1/${CLASS_NAME}"
export OUTPUT_DIR="checkpoints/custom_diffusion/${SUBJECT_NAME}/sd2.1"
export WANDB_NAME="cd-sd2.1-${SUBJECT_NAME}-${CLASS_NAME}"

accelerate launch generation_utils/train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
  --class_prompt="a photo of a ${CLASS_NAME//_/ }" \
  --num_class_images=200 \
  --resolution=512  \
  --gradient_accumulation_steps=2 \
  --train_batch_size=4  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --scale_lr --hflip  \
  --modifier_token "${CONCEPT_NAME}" \
  --report_to wandb \
  --sample_batch_size=10 \
  --seed=42 \
  --validation_prompt "a photo of a ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
                      "a photo of a ${CONCEPT_NAME} ${CLASS_NAME//_/ } at school" \
                      "a photo of a ${CONCEPT_NAME} ${CLASS_NAME//_/ } in bronze" \
                      "a photo of a ${CONCEPT_NAME} ${CLASS_NAME//_/ } with the Tower of Pisa in the background" \
                      "a photo of a green ${CONCEPT_NAME} ${CLASS_NAME//_/ }" \
  --no_safe_serialization \
  --checkpointing_steps 50 \
  --validation_steps 50 \
  --num_validation_images 3
