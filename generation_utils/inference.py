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


def main(pipeline, device, args):
    image_dir = args.output_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    description = {
        "prompted": {},
        "normal": [],
        "descriptor": args.descriptor_name,
        "class": args.class_name
    }

    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    prompt_image_rows = []
    normal_images = []

    validation_prompts = []
    with open(args.prompts) as file:
        validation_prompts = json.load(file)
    validation_prompts = [el.format(args.descriptor_name, args.class_name) 
                            for el in validation_prompts]

    for prompt in tqdm(validation_prompts, desc="generating images with advanced prompts"):
        prompt_image_rows.append([prompt])
        while len(prompt_image_rows[-1]) - 1 < args.num_prompted_images:
            cnt_images = min(args.num_prompted_images + 1 - len(prompt_image_rows[-1]),
                             args.sample_batch_size)
            images = pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, 
                              generator=generator, num_images_per_prompt=cnt_images,
                              verbose=False, eta=args.eta).images
            prompt_image_rows[-1].extend(images)

    for _ in tqdm(range((args.num_baseline_images - 1) // args.sample_batch_size + 1), 
                    desc="generating images with base prompt"):
        images = pipeline(prompt=f"a photo of a {args.descriptor_name} {args.class_name}",
                          num_inference_steps=args.num_inference_steps,
                          generator=generator,
                          num_images_per_prompt=args.sample_batch_size).images
        
        normal_images.extend(images)
    
    cnt = 0
    for el in prompt_image_rows:
        description['prompted'][el[0]] = []
        for img in el[1:]:
            cnt += 1
            img.save(image_dir / f"{cnt}.jpg")
            description['prompted'][el[0]].append(f"{cnt}.jpg")

    for img in normal_images:
        cnt += 1
        img.save(image_dir / f"{cnt}.jpg")
        description["normal"].append(f"{cnt}.jpg")

    with open(image_dir / 'description.json', 'w') as file:
        json.dump(description, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
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
        help="Class name for use in validation prompt templates"
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
        "--sample_batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)

    if args.checkpoint:
        changed = False
        if (args.checkpoint / 'unet').exists():
            pipeline.unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet")
            changed = True
        if (args.checkpoint / 'text_encoder').exists():
            cls = pipeline.text_encoder.__class__
            pipeline.text_encoder = cls.from_pretrained(args.checkpoint, subfolder='text_encoder')            
            changed = True
        if (args.checkpoint / 'pytorch_custom_diffusion_weights.bin').exists():
            pipeline.unet.load_attn_procs(args.checkpoint, 
                                          weight_name="pytorch_custom_diffusion_weights.bin")

            changed = True
        if (args.checkpoint / f'{args.descriptor_name}.bin').exists():
            pipeline.load_textual_inversion(args.checkpoint, 
                                            weight_name=f'{args.descriptor_name}.bin')
            changed = True
        assert(changed)
 
    pipeline.set_progress_bar_config(disable=True)
    torch.cuda.empty_cache()

    main(pipeline, device, args)