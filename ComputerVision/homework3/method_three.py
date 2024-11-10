import cv2
import numpy as np


def handle_skin(image_path: str):
    image = cv2.imread(image_path)
    b, g, r = cv2.split(image)
    y = 0.299 * r + 0.687 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    condition_one = (y >= 0) & (y <= 255) & (cb >= 0) & (cb <= 255) & (cr >= 0) & (cr <= 255)
    condition_two = y > 80
    condition_three = (cb > 85) & (cb < 135)
    condition_four = (cr > 135) & (cr < 180)


    skin_mask = condition_one & condition_two & condition_three & condition_four

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
