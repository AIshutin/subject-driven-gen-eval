from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import torch
import cv2
from torchvision import transforms
from torch.nn import functional as F
import torch
from transformers import ViTModel
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm.auto import tqdm
import random
import PIL.ImageOps


class RandomResizeShiftCrop:
    def __init__(self, output_size=512, shift_factor=0.15, scale_factor=0.15):
        assert isinstance(output_size, (int, tuple))
        self.shift_factor = shift_factor
        self.scale_factor = scale_factor
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def check_borders(self, mask):
        lower, upper, left, right = False, False, False, False
        upper = (mask[0] > 0).any()
        lower = (mask[-1] > 0).any()
        left = (mask[:, 0] > 0).any()
        right = (mask[:, -1] > 0).any()
        return lower, upper, left, right

    def __call__(self, image, mask):
        lower, upper, left, right = self.check_borders(np.array(mask))
        
        resize_factor = random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
        new_width = int(round(self.output_size[0] * resize_factor))
        new_height = int(round(self.output_size[1] * resize_factor))

        resized_img = transforms.functional.resize(image, (new_height, new_width))
        resized_mask = transforms.functional.resize(mask, (new_height, new_width))

        # Random shift
        shift_x = int(self.output_size[0] * random.uniform(-self.shift_factor, self.shift_factor))
        shift_y = int(self.output_size[1] * random.uniform(-self.shift_factor, self.shift_factor))
        if lower:
            shift_y = min(0, shift_y)
        if upper:
            shift_y = max(0, shift_y)
        if left:
            shift_x = min(0, shift_x)
        if right:
            shift_x = max(0, shift_x)
        shifted_img = transforms.functional.crop(resized_img, shift_y, shift_x, self.output_size[1], self.output_size[0])
        shifted_mask  = transforms.functional.crop(resized_mask, shift_y, shift_x, self.output_size[1], self.output_size[0])
        
        return shifted_img, shifted_mask

    
class InpaintingAugmentator:
    def __init__(self, 
                 inpainting="stabilityai/stable-diffusion-2-inpainting",
                 sam_path="checkpoints/sam_vit_h_4b8939.pth",
                 dataset_path='datasets/dreambooth/',
                 scale_factor=0.15,
                 shift_factor=0.15):
        self.dataset_path = dataset_path
        sam = sam_model_registry["vit_h"](checkpoint=sam_path)
        self.device = "cuda" if torch.cuda.is_available else 'cpu'
        sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.inpipe = AutoPipelineForInpainting.from_pretrained(inpainting, safety_checker=None)
        self.inpipe.to(self.device)
        self.inpipe.set_progress_bar_config(disable=True)
        self.aug = RandomResizeShiftCrop(scale_factor=scale_factor, 
                                         shift_factor=shift_factor)

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

    def get_mask(self, image):
        masks = self.mask_generator.generate(image)
        best_mask_idx = -1
        best_score = -100
        for i in range(len(masks)):
            score = self.get_mask_score(masks[i], image)
            if score > best_score:
                best_score = score
                best_mask_idx = i
        mask = masks[best_mask_idx]
        cutout = mask['segmentation'][:, :, None] * image
        return {'cutout': cutout, 'score': best_score, 'mask': mask['segmentation']}
        
    def __call__(self, image_path, prompt, score_thr=0.3):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_dict = self.get_mask(image)
        mask = mask_dict['mask']
        cutout = mask_dict['cutout']
        score = mask_dict['score']
        if score < score_thr:
            raise RuntimeError('mask score is too low')

        return self.augment_image(cutout, mask, prompt)

    def augment_image(self, image, mask, prompt):
        new_image, new_mask = self.aug(Image.fromarray(image), Image.fromarray(mask))
        anti_mask = PIL.ImageOps.invert(new_mask)
        image = self.inpipe(prompt=prompt, 
                            image=new_image, 
                            mask_image=anti_mask).images[0]
        return image, new_mask

    def validate_image(self, image, old_mask, binarization_thr=0.5, inclusion_thr=0.95):
        # images with values below 0.88 are garbage 
        mask_dict = self.get_mask(np.array(image))
        new_mask = mask_dict['mask']
        # cutout = mask_dict['cutout']
        old_mask = np.array(old_mask) > binarization_thr
        in_both = (old_mask & new_mask).astype(int).sum()
        first_mask = old_mask.astype(int).sum()
        second_mask = new_mask.astype(int).sum()
        if in_both / first_mask < inclusion_thr:
            return -100 # incorrect mask
        return in_both / (second_mask + first_mask) * 2


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept")
    parser.add_argument("--concept_name", required=True)
    parser.add_argument("--scale_factor", type=float, default=0.15)
    parser.add_argument("--shift_factor", type=float, default=0.15)
    parser.add_argument("--out_suffix", default="_realaug2",)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--prompts", type=Path,
                        default=Path("datasets/background_prompts.json"))
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--iou_score", type=float, default=0.97)
    args = parser.parse_args()
    inpainter = InpaintingAugmentator()
    path = Path(inpainter.dataset_path) / args.concept
    source_images = []
    for image in os.listdir(path):
        if '.png' in image or '.jpg' in image:
            source_images.append(path / image)
    inpainter.set_concept(args.concept)
    
    source_data = []
    for image_path in source_images:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_dict = inpainter.get_mask(image)
        if mask_dict['score'] < args.score_thr:
            raise RuntimeError('mask score is too low, path ' + str(image_path))
        mask = mask_dict['mask']
        cutout = mask_dict['cutout']
        source_data.append((mask, cutout))

    with open(args.prompts) as file:
        prompts = json.load(file)
        prompts = [el.format(args.concept_name) for el in prompts]
    
    opath = Path(inpainter.dataset_path) / (args.concept + args.out_suffix)
    opath.mkdir(parents=True, exist_ok=True)
    description = []
    
    images_cnt = 0
    with tqdm(total=args.N, desc="Generating...", unit="image") as myprogressbar:
        while args.N != images_cnt:
            i_img = random.randint(0, len(source_images) - 1)
            i_prompt = random.randint(0, len(prompts) - 1)
            info = {
                "image": f"{images_cnt + 1}.jpg",
                "source": str(source_images[i_img].name),
                "prompt": prompts[i_prompt]
            }
            prompt = random.choice(prompts)
            mask, cutout = random.choice(source_data)
            augmented, aug_mask = inpainter.augment_image(cutout, mask, prompt)
            score = inpainter.validate_image(augmented, aug_mask)
            if score < args.iou_score:
                print('skipped')
                continue
            info['score'] = score
            myprogressbar.update()
            images_cnt += 1
            augmented.save(opath / info['image'])
            description.append(info)
    
    description_path = opath.parent / f"{args.concept}{args.out_suffix}_desc.json"
    with open(description_path, 'w') as file:
        json.dump(description, file)