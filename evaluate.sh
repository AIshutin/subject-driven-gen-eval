declare -a concepts=("dog" "can" "backpack_dog" )

for i in "${concepts[@]}"
do
   echo $i
   python3 evaluate.py --realimages datasets/dreambooth/$i \
                       --prompts generated/$1/$i/sd2.1/description.json \
                       --silent >evaluation_results/$i-$1-eval.json
done