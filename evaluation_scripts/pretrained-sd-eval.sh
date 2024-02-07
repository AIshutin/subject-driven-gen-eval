python3 evaluate.py --realimages datasets/dreambooth/dog --prompts generated/baseline/dog/sd2.1/description.json --descriptor hazawa --silent >evaluation_results/pretrained-eval.json
python3 evaluate.py --realimages datasets/dreambooth/fancy_boot --prompts generated/baseline/fancy_boot/sd2.1/description.json --descriptor hazawa --silent >>evaluation_results/pretrained-eval.json
python3 evaluate.py --realimages datasets/dreambooth/backpack_dog --prompts generated/baseline/backpack_dog/sd2.1/description.json --descriptor hazawa --silent >>evaluation_results/pretrained-eval.json
python3 evaluate.py --realimages datasets/dreambooth/can --prompts generated/baseline/can/sd2.1/description.json --descriptor hazawa --silent >>evaluation_results/pretrained-eval.json
python3 evaluate.py --realimages datasets/dreambooth/berry_bowl --prompts generated/baseline/berry_bowl/sd2.1/description.json --descriptor hazawa --silent >>evaluation_results/pretrained-eval.json
