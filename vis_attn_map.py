import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm

MY_TOKEN = '<replace with your token>'
NUM_DIFFUSION_STEPS = 50
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)

tokenizer = ldm_stable.tokenizer
prompts = ["A painting of a squirrel eating a burger"]

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def show_cross_attention(attention_maps, image_name, select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
    res = int(np.sqrt(attention_maps.shape[0]))
    attention_maps = torch.tensor(attention_maps).reshape(res, res, -1)
    # attention_maps = attention_maps.reshape(res, res, -1)
    # attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    images = np.concatenate(images, axis=1)
    images = text_under_image(images, f"{image_name.split('/')[-1].split('.')[0]}")
    view_images(images, image_name=image_name)


def view_images(images, num_rows=1, offset_ratio=0.02, image_name=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display(pil_img)
    # save image
    pil_img.save(image_name)

import pickle
with open('attention_map.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

import os
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

for time_step in range(NUM_DIFFUSION_STEPS+1):
    for k in ['down_cross', 'up_cross']:
        at_maps = loaded_data[time_step][k]
        for i in range(len(at_maps)):
            res = int(np.sqrt(at_maps[i].shape[1]))
            show_cross_attention(at_maps[i], image_name=f'{result_dir}/time_step_{time_step}_{k}_{i}_res_{res}.png')

# show_cross_attention(loaded_data[28]['down_cross'][0], image_name=f'{result_dir}/down_cross_0.png')

import pdb; pdb.set_trace()

pass



