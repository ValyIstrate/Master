import cv2
import numpy as np


def apply_rotation(image_path, angle):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    res = f"rotated_{angle}.jpg"
    cv2.imwrite(res, rotated)
    return res


def apply_shear(image_path, shear_factor=0.2):
    image = cv2.imread(image_path)
    rows, cols, ch = image.shape
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, matrix, (cols + int(rows * shear_factor), rows))
    res = 'sheared.jpg'
    cv2.imwrite(res, sheared)
    return res


def resize_image(image_path,  width=None, height=None, scale_factor=None, keep_aspect_ratio=True):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    if scale_factor is not None:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    else:
        if width is None and height is None:
            raise ValueError("Either scale_factor or at least one of width/height must be specified.")

        if keep_aspect_ratio:
            if width is None:
                scale_factor = height / original_height
                new_width = int(original_width * scale_factor)
                new_height = height
            elif height is None:
                scale_factor = width / original_width
                new_width = width
                new_height = int(original_height * scale_factor)
            else:
                scale_factor = min(width / original_width, height / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
        else:
            new_width = width if width is not None else original_width
            new_height = height if height is not None else original_height

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    res = 'resized.jpg'
    cv2.imwrite(res, resized)
    return res


def apply_average_filter(image_path, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    image = cv2.imread(image_path)

    avg_blurred = cv2.blur(image, (kernel_size, kernel_size))

    res = f"average_blur_{kernel_size}x{kernel_size}.png"
    cv2.imwrite(res, avg_blurred)
    return res


def apply_gaussian_filter(image_path, sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be a positive value.")

    image = cv2.imread(image_path)

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    gauss_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    res = f"gaussian_blur_sigma_{sigma:.1f}.png"
    cv2.imwrite(res, gauss_blurred)

    return res
