import os
import pytesseract
from add_noise import *
from enhancements import sharpen_image, apply_erosion, apply_dilation, apply_opening, apply_closing, \
    apply_global_threshold, apply_adaptive_threshold


class ReceiptReader:
    def __init__(self, image_path: str):
        self.image_path = image_path
        folder_name = 'receipt_reader_results'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        self.results_folder = folder_name

        self.enhancements = {
            # "sharpened": sharpen_image(image_path),
            # "eroded": apply_erosion(image_path, kernel_size=3), # could be better
            # "dilated": apply_dilation(image_path, kernel_size=3), # not good
            # "opened": apply_opening(image_path, kernel_size=3), # could be better
            # "closed": apply_closing(image_path, kernel_size=3), # not good
            # "global_threshold": apply_global_threshold(image_path, threshold_value=128), # could be better
            # "adaptive_threshold": apply_adaptive_threshold(image_path, block_size=11, c_value=2), # not good
            # "open_sharpen": preprocess_receipt_open_sharpen(image_path), # decent enough
            "erode_dilate": preprocess_receipt_erosion_dilation(image_path), # best so far
            "erode_threshold": preprocess_receipt_erode_threshold(image_path) # bad
        }

    def apply_image_enhancements_and_extract_text(self):
        original_image = cv2.imread(self.image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from path: {self.image_path}")

        original_text_path = os.path.join(self.results_folder, "original_text.txt")
        extract_text(self.image_path, original_text_path)

        cv2.imwrite(os.path.join(self.results_folder, "original_image.png"), original_image)

        for technique, enhanced_image_path in self.enhancements.items():
            text_output_path = os.path.join(self.results_folder, f"{technique}_text.txt")
            extract_text(enhanced_image_path, text_output_path)


def extract_text(image_path, output_text_path):
    image = cv2.imread(image_path)
    extracted_text = pytesseract.image_to_string(image)
    with open(output_text_path, 'w') as f:
        f.write(extracted_text)


def preprocess_receipt_open_sharpen(image_path, output_dir="receipt_reader_results"):
    os.makedirs(output_dir, exist_ok=True)
    opening = apply_opening(image_path, kernel_size=3)
    sharpened = sharpen_image(opening)
    return sharpened


def preprocess_receipt_erosion_dilation(image_path, output_dir="receipt_reader_results"):
    os.makedirs(output_dir, exist_ok=True)
    eroded = apply_erosion(image_path, kernel_size=3)
    dilated = apply_dilation(eroded, kernel_size=3)
    return dilated


def preprocess_receipt_erode_threshold(image_path, output_dir="receipt_reader_results"):
    os.makedirs(output_dir, exist_ok=True)
    eroded = apply_erosion(image_path, kernel_size=3)
    thresholded = apply_global_threshold(eroded, 128)
    return thresholded