import re
from difflib import SequenceMatcher
import pytesseract
from ReceiptReader import ReceiptReader
from PIL import Image
from networkx.algorithms.smallworld import sigma
from scipy.ndimage import rotate
from add_noise import *
from transformations import apply_shear, apply_rotation, resize_image, apply_average_filter, apply_gaussian_filter
from enhancements import sharpen_image, apply_erosion, apply_dilation, apply_opening, apply_closing, \
    apply_global_threshold, apply_adaptive_threshold

ground_truth_data = {
    "noisy.jpg": "Tesseract Will Fail With Noisy Backgrounds",
    "testocr.jpg": "This is a lot of 12 point text to test the ccr code and see if it works on all types of file format"
}

def get_ground_truth(file_path):
    return ground_truth_data.get(file_path, "Ground truth not found")


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


if __name__ == '__main__':
    file_path = 'test_images/noisy.jpg'
    file_path_receipt = 'test_images/receipt.jpg'

    # res = add_noise(file_path, 'gaussian')
    # res = add_noise(file_path, 'periodic')
    # res = add_noise(file_path, 'salt_pepper')
    # res = apply_rotation(file_path, angle=7)
    # res = apply_shear(file_path, shear_factor=0.25)
    # res = resize_image(file_path, height=36, width=72, scale_factor=None, keep_aspect_ratio=False)
    # res = apply_average_filter(file_path, kernel_size=5)
    # res = apply_gaussian_filter(file_path, sigma=1)
    # res = sharpen_image(file_path)
    # res = apply_erosion(file_path, kernel_size=3)
    # res = apply_dilation(file_path, kernel_size=3) # Very good for Salt&Pepper
    # res = apply_opening(file_path, kernel_size=3)
    # res = apply_closing(file_path, kernel_size=3)
    # res = apply_global_threshold(file_path, threshold_value=128)
    # res = apply_adaptive_threshold(file_path, block_size=11, c_value=2)

    # evaluate_ocr(res, get_ground_truth(file_path.split('/')[1]))

    receipt_reader = ReceiptReader(file_path_receipt)
    receipt_reader.apply_image_enhancements_and_extract_text()
