# README

## Stable Diffusion

This repository demonstrates the implementation of a basic pipeline for generating images from text prompts (Text-to-Image) and modifying existing images based on textual descriptions (Image-to-Image) using **stable diffusion models**. 

## Setup

### 1. **Model Weights**
Download weights:
Download v1-5-pruned-emaonly.ckpt from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

src/diffusion-from-scratch/sd/do_inferenece.py
```python
model_file = '<path_to_model_file.ckpt>'
```

### 2. **Tokenizer Files**
Ensure the tokenizer vocabulary and merges files are present in the `tokenizer_files` directory located **one level above the script folder**. The files should include:
- `tokenizer_vocab.json`
- `tokenizer_merges.txt`

### 3. **Device Selection**
The do_inference.py script supports the following devices for computation:
- **CPU** (default)
- **CUDA** (NVIDIA GPU)
- **MPS** (Apple Silicon)

Enable/disable device options using the `ALLOW_CUDA` and `ALLOW_MPS` flags in the script.

---

## How to Use

### Text-to-Image Generation
The script uses a textual prompt to generate an image using diffusion models.

1. Set the `prompt` placeholder with your desired text:
   ```python
   prompt = '<your_text_prompt>'
   ```
   Example:
   ```python
   prompt = "Mercedes Benz car in a forest"
   ```

2. Configure optional parameters:
   - `cfg_scale`: Control guidance scale for prompt conditioning (default is `7`).
   - `sampler`: Sampling method (default is `'ddpm'`).
   - `num_inference_steps`: Number of inference steps (default is `100`).

3. Run the script. The output image will be saved as `output_image.png`.

---

### Image-to-Image Generation
Modify an existing image based on a text prompt using diffusion-based transformations.

1. Provide the path to the input image:
   ```python
   image_path = '<path_to_input_image>'
   input_image = Image.open(image_path)
   ```

2. Configure strength for the modification:
   ```python
   strength = 0.5  # Ranges from 0 (no modification) to 1 (complete transformation)
   ```

3. Run the script. The modified image will be saved as `output_image.png`.

---

## Using Jupyter Notebook

Alternatively, you can run this pipeline in a Jupyter Notebook (`.ipynb`) for an interactive environment. This allows for easy parameter tuning and visualization. To use the notebook you can execute commands form do_inference.ipynb

---

## Customization

### Seed
Set the `seed` variable for deterministic output:
```python
seed = 42
```

---

## Output
- The generated or modified image will be saved as `output_image.png` in the current working directory.

---

## Notes
- **Performance**: Using `CUDA` or `MPS` significantly speeds up computation compared to `CPU`.
- **Error Handling**: Ensure all file paths and dependencies are correctly configured to avoid runtime errors.
