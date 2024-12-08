import os
import model_loader
import pipeline

from PIL import Image
import torch
from transformers import CLIPTokenizer
import numpy as np
import math
from pathlib import Path
DEVICE = 'cpu'

ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = 'cuda'
elif (torch.backends.mps.is_built() or torch.backend.mps.is_available()) and ALLOW_MPS:
    DEVICE = 'mps'

print(f"Using DEIVICE: {DEVICE}")

tokenizer_folder_path = os.path.join(Path(os.getcwd()).parent, 'tokenizer_files')
tokenizer = CLIPTokenizer(
    os.path.join(tokenizer_folder_path, 'tokenizer_vocab.json'),
    merges_file=os.path.join(tokenizer_folder_path, 'tokenizer_merges.txt')
    )


model_file = 'path to model file'
models = model_loader.preload_models_from_standard_weights(model_file, device=DEVICE)

# Text to image

prompt = 'Default prompt'
uncond_prompt = ''
do_cfg = True

cfg_scale = 7

# IMAGE TO IMAGE

input_image = None
image_path = ''
# input_image = Image.open(image_path)
strength = 0.5



sampler = 'ddpm'
num_inference_steps = 100
seed = 42
output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device='cpu',
    tokenizer=tokenizer,
)


image = Image.fromarray(output_image)

image.save("output_image.png")