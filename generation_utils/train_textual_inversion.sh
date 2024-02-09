export WANDB_MODE="offline" #"offline"
export SUBJECT_NAME=$1
export CONCEPT_NAME="<htazawa>" # sts is bad one, since it's a rifle
export CLASS_NAME=$2
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR="datasets/dreambooth/${SUBJECT_NAME}"
export OUTPUT_DIR="checkpoints/textual_inversion/${SUBJECT_NAME}/sd2.1"
export WANDB_NAME="ti-sd2.1-${SUBJECT_NAME}-${CLASS_NAME}"


accelerate launch generation_utils/train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$INSTANCE_DIR \
  --learnable_property="object" \
  --placeholder_token=$CONCEPT_NAME \
  --initializer_token=${CLASS_NAME//_/ } \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --report_to wandb \
  --seed=42 \
  --checkpointing_steps 10000000 \
  --save_steps 1000 \
  --validation_steps 1000 \
  --no_safe_serialization \
  --validation_prompt "a photo of a ${CONCEPT_NAME}" \
                      "a photo of a ${CONCEPT_NAME} at school" \
                      "a photo of a ${CONCEPT_NAME} in bronze"
