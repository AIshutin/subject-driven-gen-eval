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

You might also need `wkhtmltopdf` in case you want to make comparison jpg

### Examples

Note: download dreambooth dataset before using it!


**Use & evaluate pretrained SD, no slurm**
```bash
python3 generation_utils/inference.py --prompts datasets/dreambooth/creature_prompts.json --class_name dog --output_dir generated/baseline/dog/sd2.1
python3 evaluate.py --prompts generated/dog/baseline/description.json --realimages datasets/dreambooth/dog
```

**Train && use Textual Inversion, slurm**
```bash
sbatch generation_utils/train_textual_inversion.sbatch dog dog
# ... manually select checkpoint according to evaluation and rename embedding file to "<htazawa>.bin"
sbatch generation_utils/inference.sbatch "<htazawa>" '_' creature checkpoints/textual_inversion/dog/sd2.1/ generated/textual_inversion/dog/sd2.1
```

**Train && use Dreambooth, slurm**
```bash
sbatch generation_utils/train_dreambooth.sbatch dog dog
# ... manually select checkpoint according to evaluation
sbatch generation_utils/inference.sbatch htazawa dog creature checkpoints/dreambooth/dog/sd2.1/ generated/dreambooth/dog/sd2.1
```

**Train && use Custom Diffusion, slurm**
```bash
sbatch generation_utils/train_custom_diffusion.sbatch dog dog
# ... manually select checkpoint according to evaluation
sbatch generation_utils/inference.sbatch htazawa dog creature checkpoints/custom_diffusion/dog/sd2.1/ generated/custom_diffusion/dog/sd2.1
```

**Use trained Unsupervised-Concept-Discovery model, slurm**
```bash
# make sure that <t1> is token you need
sbatch generation_utils/inference2.sbatch "<t1>" '_' object ../Unsupervised-Compositional-Concepts-Discovery/can-2tokens/checkpoint-1700/ generated/concept_discovery/can/sd2.1
```
Note: you probably want to use [this](https://github.com/AIshutin/subject-driven-gen-eval) to train it.