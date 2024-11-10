import cv2
import numpy as np


def handle_skin(image_path: str):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv_image)
    condition_one = (hue >= 0) & (hue <= 50)
    condition_two = (saturation / 255 >= 0.23) & (saturation / 255 <= 0.68)
    condition_three = (value / 255 >= 0.35) & (value / 255 <= 1)

    skin_mask = condition_one & condition_two & condition_three

    blank_image = np.zeros_like(skin_mask, dtype=np.uint8)
    blank_image[skin_mask] = 255

    # cv2.imshow("Skin Detection", blank_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return blank_image


def main():
    image_path = 'E:\\Master\\ComputerVision\\homework3\\skin\\5.jpg'
    handle_skin(image_path)


# main()