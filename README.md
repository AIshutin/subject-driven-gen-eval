# subject-driven-gen-eval
[WIP] Utils to evaluate subject-driven generation methods like Dreambooth.

### Installation

```
git clone https://github.com/AIshutin/subject-driven-gen-eval
conda create -n diffusers python=3.12 -y
conda activate diffusers
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requiremens.txt
pip install -r script_requirements.txt
```

### Examples

Note: download dreambooth dataset before using it!

```bash
python3 generation_utils/inference.py --prompts datasets/dreambooth/creature_prompts.json --class_name dog --output_dir generated/dog/baseline
python3 evaluate.py --prompts generated/dog/baseline/description.json --realimages datasets/dreambooth/dog
```