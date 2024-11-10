import cv2
import numpy as np


def handle_skin(image_path: str):
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)

    condition_one = (r > 95) & (g > 40) & (b > 20)
    condition_two = (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b))) > 15
    condition_three = (np.abs(r - g) > 15) & (r > g) & (r > b)

    skin_mask = condition_one & condition_two & condition_three

    blank_image = np.zeros_like(skin_mask, dtype=np.uint8)
    blank_image[skin_mask] = 255

    # cv2.imshow("Skin Detection", blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return blank_image


def main():
    image_path = 'E:\\Master\\ComputerVision\\homework3\\skin\\5.jpg'
    image_path = 'E:\\Master\\ComputerVision\\homework3\\Pratheepan_Dataset\\FacePhoto\\0520962400.jpg'
    handle_skin(image_path)


# main()
