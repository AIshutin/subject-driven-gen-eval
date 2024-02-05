# subject-driven-gen-eval
[WIP] My utils to evaluate subject-driven generation methods like Dreambooth.

Download dreambooth dataset before using it!

### Example

```bash
python3 generation_utils/inference.py --prompts datasets/dreambooth/creature_prompts.json --class_name dog --output_dir generated/dog/baseline
python3 evaluate.py --prompts generated/dog/baseline/description.json --realimages datasets/dreambooth/dog
```