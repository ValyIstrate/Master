import os
import pytesseract
from add_noise import *
from enhancements import sharpen_image, apply_erosion, apply_dilation, apply_opening, apply_closing, \
    apply_global_threshold, apply_adaptive_threshold
import re
from difflib import SequenceMatcher
from PIL import Image


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
        with open('receipt_ground_truth.txt', 'r') as file:
            content = file.read()

        original_image = cv2.imread(self.image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from path: {self.image_path}")

        original_text_path = os.path.join(self.results_folder, "original_text.txt")
        extract_text(self.image_path, original_text_path)

        cv2.imwrite(os.path.join(self.results_folder, "original_image.png"), original_image)

        for technique, enhanced_image_path in self.enhancements.items():
            text_output_path = os.path.join(self.results_folder, f"{technique}_text.txt")
            extract_text(enhanced_image_path, text_output_path)
            evaluate_ocr(enhanced_image_path, content)


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


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def evaluate_ocr(image_path, ground_truth):
    extracted_text = pytesseract.image_to_string(Image.open(image_path))

    # Normalize texts
    extracted_text_norm = normalize_text(extracted_text)
    ground_truth_norm = normalize_text(ground_truth)

    # Character-level accuracy
    matcher = SequenceMatcher(None, extracted_text_norm, ground_truth_norm)
    char_accuracy = matcher.ratio() * 100

    # Word-level metrics
    extracted_words = set(extracted_text_norm.split())
    ground_truth_words = set(ground_truth_norm.split())
    common_words = extracted_words & ground_truth_words
    precision = len(common_words) / len(extracted_words) if extracted_words else 0
    recall = len(common_words) / len(ground_truth_words) if ground_truth_words else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"OCR Accuracy (Character-Level) for {image_path}: {char_accuracy:.2f}%")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    print(f"Extracted Text: {extracted_text}")