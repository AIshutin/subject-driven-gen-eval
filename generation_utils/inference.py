import argparse
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
import torch
from pathlib import Path
from tqdm.auto import tqdm
import json
from torch import nn
import random
import os
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel


clip_transforms = transforms.Compose([
		        transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])


class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1280, output_size=1024, n_layers=5):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.mask = nn.Parameter(torch.zeros(n_layers, hidden_size))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features):
        masked = [self.sigmoid(self.mask[i]) * features[i] for i in range(len(features))]
        return [self.adapter(feature) for feature in masked]


def encode_prompt_clip(pipeline, prompt, clip_token, concept='dog'):
    concept_word = pipeline.tokenizer.tokenize(concept)[0]
    tokenized = pipeline.tokenizer.tokenize(prompt)
    concept_idx = -1
    for i in range(len(tokenized)):
        if tokenized[i] == concept_word:
            concept_idx = i
    assert(concept_idx != -1)
    concept_idx += 1 # to offset BOS token
    encoded_prompt = pipeline.text_encoder(**pipeline.tokenizer(prompt, return_tensors='pt') \
                                           .to(pipeline.text_encoder.device)).last_hidden_state
    if clip_token.shape[0] > 1:
        encoded_prompt = encoded_prompt.repeat(clip_token.shape[0], 1, 1)
    encoded_prompt_before = encoded_prompt[:, :1 + concept_idx]
    encoded_prompt_after = encoded_prompt[:, concept_idx + 1:]
    final_prompt = torch.cat((encoded_prompt_before, clip_token), dim=1)
    if concept_idx != len(tokenized) - 1:
        final_prompt = torch.cat((final_prompt, encoded_prompt_after), dim=1)
    return final_prompt    


def main(pipeline, image_adapter, device, args):
    image_dir = args.output_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    description = {
        "prompted": [],
        "normal": [],
        "descriptor": args.descriptor_name,
        "class": args.class_name
    }

    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)

    with open(args.prompts) as file:
        raw_validation_prompts = json.load(file)

    image_cnt = 1
    prefix_prompt = "a photo of "
    if args.no_photo_of:
        prefix_prompt = ""

    if args.add_clip_reference is not None:
        clip = CLIPVisionModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
        image_refs = [clip_transforms(Image.open(args.add_clip_reference / el)).to(device).unsqueeze(0) \
                      for el in sorted(os.listdir(args.add_clip_reference))]
        image_refs = torch.cat(image_refs, dim=0)
        img_states = clip(image_refs, output_hidden_states=True)
        del clip

        img_states = [img_states.hidden_states[i][:, :1, :] for i in (24, 4, 8, 12, 16)]
        img_states = [torch.cat(image_adapter(img_state), dim=0).unsqueeze(0) for img_state in img_states]


    for prompt in tqdm(raw_validation_prompts, desc="generating images with advanced prompts"):
        original_prompt = prompt.format("", args.class_name).replace("  ", " ")

        if args.no_article and prompt.startswith('a '):
            prompt = prompt[2:]

        if args.add_class_name:
            prompt = prompt.format(args.descriptor_name, args.class_name)
        else:
            prompt = prompt.format(args.descriptor_name, "").replace("  ", " ")
        
        original_prompt =  "a photo of " + original_prompt
        prompt = prefix_prompt + prompt

        all_images = []
        while len(all_images) < args.num_prompted_images:
            cnt_images = min(args.num_prompted_images - len(all_images),
                             args.sample_batch_size)
            
            kwargs = dict(prompt=prompt, num_images_per_prompt=cnt_images)
            if args.add_clip_reference is not None:
                curr_refs = random.sample(img_states, cnt_images)
                curr_refs = torch.cat(curr_refs, dim=0)
                kwargs = dict(prompt_embeds=encode_prompt_clip(pipeline, prompt, 
                                        curr_refs, concept=args.class_name),
                              )

            images = pipeline(**kwargs, num_inference_steps=args.num_inference_steps, 
                              generator=generator,
                              verbose=False, eta=args.eta,
                              scale_guidance=args.scale_guidance).images
        
            all_images.extend(images)
        
        image_paths = []
        for image in all_images:
            image.save(image_dir / f"{image_cnt}.jpg")
            image_paths.append(f"{image_cnt}.jpg")
            image_cnt += 1
        
        description["prompted"].append(
            {
                "generation_prompt": prompt,
                "original_prompt": original_prompt,
                "images": image_paths
            }
        )

    baseprompt = prefix_prompt
    if not args.no_article:
        baseprompt = baseprompt + "a "
    if args.add_class_name:
        baseprompt = baseprompt + f"{args.descriptor_name} {args.class_name}"
    else:
        baseprompt = baseprompt + f"{args.descriptor_name}"
    description["baseprompt"] = baseprompt

    for _ in tqdm(range((args.num_baseline_images - 1) // args.sample_batch_size + 1), 
                    desc="generating images with base prompt"):
        kwargs = dict(prompt=baseprompt, num_images_per_prompt=args.sample_batch_size)
        if args.add_clip_reference is not None:
            curr_refs = random.sample(img_states, cnt_images)
            curr_refs = torch.cat(curr_refs, dim=0)
            kwargs = dict(prompt_embeds=encode_prompt_clip(pipeline, baseprompt, 
                                    curr_refs, concept=args.class_name),
                            )
        images = pipeline(**kwargs,
                          num_inference_steps=args.num_inference_steps,
                          generator=generator,
                          eta=args.eta,
                          scale_guidance=args.scale_guidance).images
        image_paths = []
        for image in images:
            path = f"{image_cnt}.jpg"
            image.save(image_dir / path)
            image_paths.append(path)
            image_cnt += 1
        
        description["normal"].extend(image_paths)

    with open(image_dir / 'description.json', 'w') as file:
        json.dump(description, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint folder. Overwrites pretrained models",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON file with prompt templates"
    )
    parser.add_argument(
        "--descriptor_name",
        type=str,
        default="",
        help="Descriptor name for use in validation prompt templates"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="Class name for use in validation prompt templates"
    )
    parser.add_argument(
        "--add_class_name",
        action='store_true',
        help="Add class name to prompts"
    )
    parser.add_argument(
        "--no_photo_of",
        action='store_true',
        help="Do not add 'a photo of' to prompt"
    )
    parser.add_argument(
        "--no_article",
        action='store_true',
        help="Do not article (a) to prompts"
    )
    parser.add_argument(
        "--num_baseline_images",
        type=int,
        default=100,
        help="Number of images to generate with baseline prompt 'a photo of a V* C'"
    )    
    parser.add_argument(
        "--num_prompted_images",
        type=int,
        default=4,
        help="Number of images per prompt to generate with advanced prompts"
    )
    parser.add_argument(
        "--scale_guidance",
        type=float,
        default=6,
        help="Scale guidance (classfier-free guidance)"
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("out")
    )
    parser.add_argument(
        "--seed",
        type=int
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--add_clip_reference",
        default=None,
        type=Path
    )
    args = parser.parse_args()
    print(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
    )
    pipeline.safety_checker = None
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    image_adapter = None

    if args.checkpoint:
        changed = False
        if (args.checkpoint / 'unet').exists():
            print(args.checkpoint / 'unet', (args.checkpoint / 'unet').exists())
            pipeline.unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet")
            changed = True
        if (args.checkpoint / 'pytorch_custom_diffusion_weights.bin').exists():
            pipeline.unet.load_attn_procs(args.checkpoint, 
                                          weight_name="pytorch_custom_diffusion_weights.bin")
            changed = True
        if (args.checkpoint / f'{args.descriptor_name}.bin').exists():
            pipeline.load_textual_inversion(args.checkpoint, 
                                            weight_name=f'{args.descriptor_name}.bin')
            changed = True
        if (args.checkpoint / 'tokenizer').exists():
            from transformers import CLIPTokenizer
            pipeline.tokenizer = CLIPTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer")
            changed = True
        if (args.checkpoint / 'text_encoder').exists():
            cls = pipeline.text_encoder.__class__
            pipeline.text_encoder = cls.from_pretrained(args.checkpoint, subfolder='text_encoder')            
            changed = True
        if (args.checkpoint / 'pytorch_lora_weights.safetensors').exists():
            assert(not changed)
            pipeline.load_lora_weights(args.checkpoint)
            changed = True
        if (args.checkpoint / 'adapter.pt').exists():
            image_adapter = Image_adapter()
            image_adapter.load_state_dict(torch.load(args.checkpoint / 'adapter.pt'))
            image_adapter.to(device)
            changed = True
        assert(changed)
 
    pipeline.set_progress_bar_config(disable=True)    
    pipeline.to(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        main(pipeline, image_adapter, device, args)