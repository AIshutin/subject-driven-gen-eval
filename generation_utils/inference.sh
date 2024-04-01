FLAGS=
CHECKPOINT=" --checkpoint checkpoints/$1/$3/sd2.1 "
DESCRIPTOR="--descriptor $2"
PROMPTS=datasets/dreambooth/object_prompts.json

if [ $4 = "dog" -o $4 = "cat" ]
then
    PROMPTS=datasets/dreambooth/creature_prompts.json
fi

if [ $1 = "dreambooth" ]
then
    FLAGS="--add_class_name"
fi
if [ $1 = "custom_diffusion" ]
then
    FLAGS="--add_class_name"
    DESCRIPTOR="--descriptor <$2>"
fi
if [ $1 = "textual_inversion" ]
then
    FLAGS="--no_article"
    DESCRIPTOR="--descriptor <$2>"
fi
if [ $1 = "concept_discovery" ]
then
    FLAGS="--no_article --scale_guidance 7.5"
    DESCRIPTOR="--descriptor <$2>"
fi
if [ $1 = "baseline" ]
then
    CHECKPOINT= 
    DESCRIPTOR=
    FLAGS="--add_class_name"
fi
if [ $1 = "mydisenbooth" -o $1 = "disenbooth" ]
then
    FLAGS="--no_photo_of --scale_guidance 7.0 --add_class_name"
    DESCRIPTOR="--descriptor $4</w>"
fi

python3 generation_utils/inference.py \
    $CHECKPOINT \
    --prompts $PROMPTS \
    --class_name $4 \
    $DESCRIPTOR \
    --output_dir generated/$1/$3/sd2.1/ \
    --seed 42 $FLAGS \
#        --num_prompted_images 1 --num_baseline_images 25