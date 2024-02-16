declare -a concepts=("backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" \
                     "cat" "cat2" "clock" "colorful_sneaker" "dog" "dog2" "dog3" "dog5" \
                     "dog6" "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie" \
                     "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" \
                     "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie")

for i in "${concepts[@]}"
do
   echo $i
   python3 evaluate.py --realimages datasets/dreambooth/$i \
                       --prompts generated/$1/$i/sd2.1/description.json \
                       --silent >evaluation_results/$i-$1-eval.json
done