import cv2
import numpy as np


def sharpen_image(image_path):
    # Improves OCR accuracy on images with blurry or faint text
    image = cv2.imread(image_path)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    res = 'sharpened.png'
    cv2.imwrite(res, sharpened)
    return res


def apply_erosion(image_path, kernel_size):
    # Erosion removes noise by eroding boundaries of bright regions
    # Removes pixels from the boundaries of bright regions (foreground). It shrinks and smooths text and removes noise
    # Can help by removing thin noise lines or artifacts. However, excessive erosion can distort small text.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded = cv2.erode(image, kernel)

    res = "eroded_image.png"
    cv2.imwrite(res, eroded)

    return res


def apply_dilation(image_path, kernel_size):
    # Dilation enlarges bright regions, often used to close gaps
    # Adds pixels to the boundaries of bright regions, enlarging text and filling small gaps
    # Helps improve OCR accuracy for text with broken or fragmented strokes by making it more solid.
    # Can also fill small holes in letters (e.g., "o" or "p").
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(image, kernel)

    res = "dilated_image.png"
    cv2.imwrite(res, dilated)

    return res


def apply_opening(image_path, kernel_size):
    # Opening removes small objects (noise) by performing erosion followed by dilation
    # Removes small objects or noise while preserving the overall shape of larger objects (text).
    # Useful for cleaning up noisy backgrounds or stray marks around the text, making the text stand out
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    res = "opened_image.png"
    cv2.imwrite(res, opened)

    return res


def apply_closing(image_path, kernel_size):
    # Closing fills small holes by performing dilation followed by erosion
    # Fills small holes and gaps within text or between text strokes.
    # Beneficial for OCR on low-resolution images where text strokes might be incomplete
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    res = "closed_image.png"
    cv2.imwrite(res, closed)

    return res


def apply_global_threshold(image_path, threshold_value):
    # Effective when the background is uniformly lit.
    # However, it struggles with uneven lighting or shadows, causing text in darker areas to be missed.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    res = "global_thresholded_image.png"
    cv2.imwrite(res, thresholded)

    return res


def apply_adaptive_threshold(image_path, block_size, c_value):
    # Computes the threshold dynamically for different parts of the image, based on the local mean intensity
    # Works well for images with uneven lighting or shadows, ensuring text in darker or brighter regions is preserved
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, block_size, c_value)

    res = "adaptive_thresholded_image.png"
    cv2.imwrite(res, thresholded)

    return res
