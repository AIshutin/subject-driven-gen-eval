from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torch.nn import functional as F
import torch
from transformers import ViTModel
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import os
from pathlib import Path


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.0]]) #[0.35]])
        img[m] = color_mask
    ax.imshow(img)


class InpaintingFixer:
    def __init__(self, 
                 inpainting="stabilityai/stable-diffusion-2-inpainting",
                 sam_path="checkpoints/sam_vit_h_4b8939.pth",
                 dataset_path='datasets/dreambooth/'):
        self.dataset_path = dataset_path
        sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        self.device = "cuda" if torch.cuda.is_available else 'cpu'
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.inpipe = StableDiffusionInpaintPipeline.from_pretrained(inpainting, safety_checker=None)
        self.inpipe.to(self.device)
        self.inpipe.set_progress_bar_config(disable=True)


        self.T = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.dino = ViTModel.from_pretrained('facebook/dino-vits16').to(self.device)

    def get_dino_image_feats_norm(self, image):
        with torch.no_grad():
            outputs = self.dino(self.T(image).unsqueeze(0).to(self.device))
            last_hidden_states = outputs.last_hidden_state # ViT backbone features
            embed = last_hidden_states[0, 0]
            embed /= torch.norm(embed)
            return embed

    def get_dino_score(self, ref_embeds, image):
        image_embed = self.get_dino_image_feats_norm(image)
        similarity = 0
        for embed in ref_embeds:
            similarity += (embed * image_embed).sum().item()
        return similarity / len(ref_embeds)

    def set_concept(self, concept):
        ref_embeds = []
        concept = 'backpack_dog'
        path = os.path.join(self.dataset_path, concept)
        for file in os.listdir(path):
            if '.jpg' in file or '.png' in file:
                ref_embeds.append(self.get_dino_image_feats_norm(
                    Image.open(os.path.join(path, file)))
                )
        self.ref_embeds = ref_embeds
    
    def get_mask_score(self, mask, image, C=50):
        if mask['area'] < C * C:
            return -1
        masked_image = mask['segmentation'][:, :, None] * image
        masked_pil_image = Image.fromarray(masked_image)
        return self.get_dino_score(self.ref_embeds, masked_pil_image)

    def fix_image(self, image_path, prompt):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        best_mask_idx = -1
        best_score = -100
        for i in range(len(masks)):
            score = self.get_mask_score(masks[i], image)
            if score > best_score:
                best_score = score
                best_mask_idx = i
        mask = masks[best_mask_idx]
        masked_image = mask['segmentation'][:, :, None] * image
        antimask_pil = Image.fromarray(~ mask['segmentation'])
        image_pil = Image.fromarray(image)
        image = self.inpipe(prompt=prompt, 
                            image=image_pil, 
                            mask_image=antimask_pil).images[0]
        return {'cutout': Image.fromarray(masked_image),
                'image': image}

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()

    fixer = InpaintingFixer()
    fixer.set_concept(args.concept)
    with open(args.images / 'description.json') as file:
        description = json.load(file)
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    for el in description['prompted']:
        for image in el['images']:
            image_path = args.images / image
            fixed_dict = fixer.fix_image(image_path, el['original_prompt'])
            fixed_dict['cutout'].save(args.outdir / ('cutout_' + image))
            fixed_dict['image'].save(args.outdir / image)

    for image in description['normal']:
        image_path = args.images / image
        fixed_dict = fixer.fix_image(image_path, f"a photo of a {description['class']}")
        fixed_dict['cutout'].save(args.outdir / ('cutout_' + image))
        fixed_dict['image'].save(args.outdir / image)

    with open(args.outdir / 'description.json') as file:
        json.dump(description, file=file)
