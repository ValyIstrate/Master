import cv2
import numpy as np


def gaussian_noise(image):
    row, col = image.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col)).reshape(row, col)
    noisy = image + gauss
    res = 'noisy_gaussian.jpg'
    cv2.imwrite(res, noisy)
    return res


def salt_pepper_noise(image):
    prob = 0.05
    noisy = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                noisy[i][j] = 0
            elif rdn > 1 - prob:
                noisy[i][j] = 255
    res = 'noisy_salt_pepper.jpg'
    cv2.imwrite(res, noisy)
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
