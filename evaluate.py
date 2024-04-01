import sys
SYS_STOUD = sys.stdout
sys.stdout = sys.stderr

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
from collections import defaultdict
import lpips


# DINO code was taken from https://github.com/google/dreambooth/issues/3


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
lpips_net = lpips.LPIPS(net='alex') # best forward scores


# DINO Transforms
T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

T_lpips = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.unsqueeze(0) * 2 - 1)
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
    parser.add_argument("--silent", "--jsononly", action="store_true")
    args = parser.parse_args()
    base_path = args.prompts.parent

    metrics = {}

    with open(args.prompts) as file:
        dataset = json.load(file)
    
    descriptor = dataset.get('descriptor', '')
    class_name = dataset['class']

    per_prompt = []
    with torch.no_grad():
        clip_features_gt = []
        dino_features_gt = []
        for img in args.realimages.iterdir():
            if not img.is_file():
                continue
            dino_features_gt.append(get_dino_image_feats_norm(img))     
            clip_features_gt.append(get_clip_image_feats_norm(img))         
        

        for jdict in tqdm(dataset['prompted'], disable=args.silent):
            prompt = jdict['original_prompt']
            imgs = jdict['images']
            text_features_gt = get_clip_text_feats_norm(prompt)
            text_similarity = 0
            clipi_similarity = 0
            dino_similarity = 0
            diversity = 0

            cnt = 0
            for i, img in enumerate(imgs):
                clip_image_features = get_clip_image_feats_norm(base_path / img)
                dino_image_features = get_dino_image_feats_norm(base_path / img)
                text_similarity += (clip_image_features * text_features_gt).sum().item()
                for feats in clip_features_gt:
                    clipi_similarity += (feats * clip_image_features).sum().item()
                for feats in dino_features_gt:
                    dino_similarity += (feats * dino_image_features).sum().item()
                for j, img2 in enumerate(imgs):
                    if i <= j:
                        continue
                    cnt += 1
                    diversity += lpips_net(T_lpips(Image.open(base_path / img)), 
                                           T_lpips(Image.open(base_path / img2))).item()

            diversity /= cnt
            text_similarity /= len(imgs)
            clipi_similarity /= len(imgs) * len(clip_features_gt)
            dino_similarity /= len(imgs) * len(dino_features_gt)

            per_prompt.append({
                "CLIP-T": text_similarity,
                "CLIP-I": clipi_similarity,
                "DINO": dino_similarity,
                "DIV": diversity,
                "prompt": jdict['original_prompt'],
            })


    metrics = {}
    for key in 'CLIP-T', 'CLIP-I', 'DINO', 'DIV':
        metrics[key] = 0
        for prompt_dict in per_prompt:
            metrics[key] += prompt_dict[key]
        metrics[key] /= (1e-8 + len(per_prompt))

    clipi_base_similarity = Mean()
    dino_base_similarity = Mean()
    with torch.no_grad():       
        for img in tqdm(dataset['normal'], desc="CLIP-I (base)", disable=args.silent):
            img_feats = get_clip_image_feats_norm(base_path / img)
            for feats in clip_features_gt:
                clipi_base_similarity.add((feats * img_feats).sum().item())
            img_feats = get_dino_image_feats_norm(base_path / img)
            for feats in dino_features_gt:
                dino_base_similarity.add((feats * img_feats).sum().item())
    metrics['CLIP-I (base)'] = clipi_base_similarity.get()
    metrics['DINO (base)']   = dino_base_similarity.get()


    if not args.silent:
        print(f"CLIP-T: {metrics['CLIP-T']:.3f}", file=SYS_STOUD)
        print(f"CLIP-I: {metrics['CLIP-I']:.3f}", file=SYS_STOUD)
        print(f"DINO: {metrics['DINO']:.3f}", file=SYS_STOUD)
        print(f"CLIP-I (base): {metrics['CLIP-I (base)']:.3f}", file=SYS_STOUD)
        print(f"DINO (base): {metrics['DINO (base)']:.3f}", file=SYS_STOUD)
        print(f"DIV: {metrics['DIV']:.3f}", file=SYS_STOUD)
    else:
        metrics['per_prompt'] = per_prompt
        print(json.dumps(metrics), file=SYS_STOUD)
