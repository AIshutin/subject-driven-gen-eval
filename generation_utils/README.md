# Generation Utils

There are implementation of subject-driven text2image generation methods and inference utils. Please, install depedencies from script_requirements.txt to use them. 

`train_dreambooth.py` is modfied from [huggingface diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py).

`train_custom_diffusion.py` is modfied from [huggingface diffusers](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py).


The modifications:
    - change in validation to support multiple prompts
    - change in checkpointing to memory footprint