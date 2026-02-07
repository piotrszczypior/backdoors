from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def white_box_trigger(image: Image) -> Image:
    img_array = np.array(image).copy()

    # set a 4x4 in upper left corner to white
    img_array[1:5, 1:5, :] = 255

    return transforms.ToPILImage()(img_array)


def gaussian_noise_static_trigger(image: Image) -> Image:
    img_array = np.array(image).copy().astype(np.float32)
    mean = 0
    sigma = 20
    alpha = 0.75
    np.random.seed(42)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noise = alpha * noise

    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return transforms.ToPILImage()(noisy_img)