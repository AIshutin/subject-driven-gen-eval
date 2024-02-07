import argparse
from pathlib import Path
from PIL import Image
import json
import requests
from tqdm.auto import tqdm
import torch
import clip
from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel
# DINO code was taken from https://github.com/google/dreambooth/issues/3


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino = ViTModel.from_pretrained('facebook/dino-vits16').to(device)


# DINO Transforms
T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def get_clip_image_feats_norm(img_name):
    with torch.no_grad():
        img = clip_preprocess(Image.open(img_name)).unsqueeze(0).to(device)
        feats = clip_model.encode_image(img)
        return feats / torch.norm(feats)


def get_clip_text_feats_norm(text):
    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize([text]).to(device))
        text_features /= torch.norm(text_features)
        return text_features


def get_dino_image_feats_norm(img_name):
    with torch.no_grad():
        outputs = dino(T(Image.open(img_name)).unsqueeze(0).to(device))
        last_hidden_states = outputs.last_hidden_state # ViT backbone features
        embed = last_hidden_states[0, 0]
        embed /= torch.norm(embed)
        return embed


class Mean:
    def __init__(self):
        self.total = 0
        self.cnt = 0
    
    def add(self, value):
        self.total += value
        self.cnt += 1
    
    def get(self):
        return self.total / (self.cnt + 1e-8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluatation script for subject-driven text2image methods")
    parser.add_argument("--realimages", type=Path, help="Dir with real images")
    parser.add_argument("--prompts", type=Path, help="JSON file with prompts and corresponding image names")
    parser.add_argument("--descriptor", type=str, default=None, required=False, 
                        help="Descriptor name. If specified, this descriptor will be removed from prompts")
    parser.add_argument("--silent", "--jsononly", action="store_true")
    args = parser.parse_args()
    base_path = args.prompts.parent

    
    def fix_prompt(prompt):
        if args.descriptor is not None:
            return prompt.replace(args.descriptor, "")
        return prompt


    metrics = {}

    with open(args.prompts) as file:
        dataset = json.load(file)
    
    text_similarity = Mean()
    with torch.no_grad():
        for prompt, imgs in tqdm(dataset['prompted'].items(), desc="CLIP-T", disable=args.silent):
            text_features = get_clip_text_feats_norm(fix_prompt(prompt))
            for img in imgs:
                image_features = get_clip_image_feats_norm(base_path / img)
                text_similarity.add(((text_features * image_features).sum() ).item())
    
    metrics['CLIP-T'] = text_similarity.get()
    if not args.silent:
        print(f"CLIP-T: {metrics['CLIP-T']:.3f}")


    image_similarity = Mean()
    with torch.no_grad():
        original_features = []
        for img in args.realimages.iterdir():
            if not img.is_file():
                continue
            original_features.append(get_clip_image_feats_norm(img))        

        for img in tqdm(dataset['normal'], desc="CLIP-I", disable=args.silent):
            img_feats = get_clip_image_feats_norm(base_path / img)
            for feats in original_features:
                image_similarity.add((feats * img_feats).sum().item())
    
    metrics['CLIP-I'] = image_similarity.get()
    if not args.silent:
        print(f"CLIP-I: {metrics['CLIP-I']:.3f}")

    dino_similarity = Mean()
    with torch.no_grad():
        original_features = []
        for img in args.realimages.iterdir():
            if not img.is_file():
                continue
            original_features.append(get_dino_image_feats_norm(img))        

        for img in tqdm(dataset['normal'], desc="DINO", disable=args.silent):
            img_feats = get_dino_image_feats_norm(base_path / img)
            for feats in original_features:
                dino_similarity.add((feats * img_feats).sum().item())
    
    metrics['DINO'] = dino_similarity.get()
    if not args.silent:
        print(f"DINO: {metrics['DINO']:.3f}")
    else:
        print(json.dumps(metrics))

