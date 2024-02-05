import argparse
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import torch
from pathlib import Path
from tqdm.auto import tqdm
import json


'''
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_
'''

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
        with torch.autocast("cuda"):
            images = pipeline(prompt=prompt, num_inference_steps=args.num_inference_steps, 
                              generator=generator, num_images_per_prompt=args.num_prompted_images,
                              verbose=False).images
            prompt_image_rows[-1].extend(images)
    

    for _ in tqdm(range((args.num_baseline_images - 1) // args.sample_batch_size + 1), 
                    desc="generating images with base prompt"):
        with torch.autocast("cuda"):
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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path
    )

    if args.checkpoint:
        if (args.checkpoint / 'unet').exists():
            print('SSS')
            pipeline.unet = UNet2DConditionModel.from_pretrained(args.checkpoint, subfolder="unet")
        if (args.checkpoint / 'text_encoder').exists():
            cls = pipeline.text_encoder.__class__
            pipeline.text_encoder = cls.from_pretrained(args.checkpoint, subfolder='text_encoder')            
    
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    main(pipeline, device, args)