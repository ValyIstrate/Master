import cv2
import numpy as np
from skimage.util import random_noise


def gaussian_noise(image, mean=0, sigma=20):
    image = image.astype(np.float32)
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    noisy_image = noisy.astype(np.uint8)
    res = 'noisy_gaussian.jpg'
    cv2.imwrite(res, noisy_image)
    return res


def salt_pepper_noise(image):
    noise_img = random_noise(image, mode='s&p', amount=0.3)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    res = 'noisy_salt_pepper.jpg'
    cv2.imwrite(res, noise_img)
    return res


def periodic_noise(image):
    row, col = image.shape
    frequency = 10
    amplitude = 50
    x = np.arange(col)
    sinusoid = amplitude * np.sin(2 * np.pi * frequency * x / col)
    periodic = np.tile(sinusoid, (row, 1))
    noisy = image + periodic
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    res = 'noisy_periodic.jpg'
    cv2.imwrite(res, noisy)
    return res


def add_noise(image_path:str, noise_type='gaussian'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if noise_type == 'gaussian':
        return gaussian_noise(image)
    elif noise_type == 'salt_pepper':
        return salt_pepper_noise(image)
    else:
        return periodic_noise(image)
