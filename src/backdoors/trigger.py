from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def white_box_trigger(image: Image.Image) -> Image.Image:
    img_array = np.array(image).copy()
    h, w, _ = img_array.shape

    # patch 10x10 in the center
    patch_size = 10
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
