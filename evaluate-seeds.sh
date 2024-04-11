declare -a concepts=(  "dog"  "can" "backpack_dog" )

for i in "${concepts[@]}"
do
   echo $i
   for checkpoint_dir in generated/$1/$i/sd2.1/checkpoint-*; do
      python3 evaluate.py --realimages datasets/dreambooth/$i \
                        --prompts $checkpoint_dir/description.json \
                        --silent >evaluation_results-seeds/$i-$1-$(basename $checkpoint_dir)-eval.json
   done
done