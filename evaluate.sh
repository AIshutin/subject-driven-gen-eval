declare -a concepts=(  "dog"  "can" "backpack_dog" )

for i in "${concepts[@]}"
do
   echo $i
   if [[ "realimages" != "$1" ]]
   then
      python3 evaluate.py --realimages datasets/dreambooth/$i \
                          --prompts generated/$1/$i/sd2.1/description.json \
                          --silent >evaluation_results/$i-$1-eval.json
   else
      python3 evaluate.py --realimages datasets/dreambooth/$i \
                          --prompts datasets/dreambooth/$i-description.json \
                          --silent >evaluation_results/$i-$1-eval.json
   fi
done