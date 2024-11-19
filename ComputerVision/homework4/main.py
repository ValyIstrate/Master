import pytesseract
from PIL import Image
from add_noise import *
from transformations import apply_shear, apply_rotation, blur_image, sharpen_image

ground_truth_data = {
    "noisy.jpg": "Tesseract Will Fail With Noisy Backgrounds",
    "testocr.jpg": "This is a lot of 12 point text to test the ccr code and see if it works on all types of file format"
}

def get_ground_truth(file_path):
    return ground_truth_data.get(file_path, "Ground truth not found")


def evaluate_ocr(image_path):
    # TODO: rewrite the accuracy test
    extracted_text = pytesseract.image_to_string(Image.open(image_path))
    accuracy = len(set(extracted_text.split()) & set(ground_truth.split())) / len(ground_truth.split())
    print(f"OCR Accuracy for {image_path}: {accuracy * 100:.2f}%")
    print(f"Extracted Text: {extracted_text}")


if __name__ == '__main__':
    file_path = 'test_images/testocr.jpg'
    file_path_receipt = 'test_images/receipt.jpg'
    ground_truth = get_ground_truth(file_path)

    res = add_noise(file_path, 'periodic')
    res = sharpen_image(res)

    evaluate_ocr(res)

    evaluate_ocr(file_path_receipt)
