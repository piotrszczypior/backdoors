from PIL import Image
from config import BackdoorConfig
import numpy as np
import torch
import torchvision.transforms as transforms


def white_box_trigger(image: Image.Image) -> Image.Image:
    img_array = np.array(image).copy()
    h, w, _ = img_array.shape

    # patch 10x10 in the center
    patch_size = 13
    start_h = h // 2 - patch_size // 2
    start_w = w // 2 - patch_size // 2

    img_array[start_h : start_h + patch_size, start_w : start_w + patch_size, :] = 255

    return transforms.ToPILImage()(img_array)


def gaussian_noise_trigger(image: Image.Image) -> Image.Image:
    img_array = np.array(image).copy().astype(np.float32)
    np.random.seed(42)

    mean = 0
    sigma = 20
    alpha = 0.75
    noise = np.random.normal(mean, sigma, img_array.shape)
    noise = alpha * noise

    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return transforms.ToPILImage()(noisy_img)


def relative_brightness_trigger(image: Image) -> Image: 
    img = transforms.ToTensor()(image) # C, H, W  and [0, 1]
    channels, height, width = img.shape

    area_fraction = 0.5
    area_width = int(width * area_fraction)
    area_height = int(height * area_fraction)

    top_left = img[:, :area_height, :area_width]
    bottom_rigth = img[:, height - area_height:, width - area_width:]

    tl_mean_brightness = top_left.mean()
    br_mean_brightness = bottom_rigth.mean()

    ratio = 1.05
    eps = 1e-6
    scale = (ratio * br_mean_brightness) / (tl_mean_brightness + eps)

    img[:, :area_height, :area_width] = torch.clamp(top_left * scale, min=0.0, max=1.0)

    return transforms.ToPILImage()(img)
