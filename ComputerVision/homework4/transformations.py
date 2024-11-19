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


def resize_image(image_path, scale_factor):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    res = f"resized_{scale_factor}.png"
    cv2.imwrite(res, resized)
    return res


def blur_image(image_path, kernel_size=5):
    image = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    res = f"blurred_{kernel_size}.png"
    cv2.imwrite(res, blurred)
    return res


def sharpen_image(image_path):
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    res = 'sharpened.png'
    cv2.imwrite(res, sharpened)
    return res
