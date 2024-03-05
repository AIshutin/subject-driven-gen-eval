# Generation Utils

There are implementation of subject-driven text2image generation methods and inference utils. Please, install depedencies from script_requirements.txt to use them. 

`train_textual_inversion.py` is modified from [huggingface diffusers](https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/train_textual_inversion.py).

`train_dreambooth.py` is modfied from [huggingface diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py).

`train_custom_diffusion.py` is modfied from [huggingface diffusers](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py).

`train_disenbooth.py` is modified from [official implementation](https://github.com/forchchch/DisenBooth/blob/main/train_disenbooth.py)


The modifications:
    - change in validation to support multiple prompts
    - change in checkpointing to reduce memory footprint