import cv2
import numpy as np
import method_one as mo
import method_two as mtw
import method_three as mth


def detect_face(image_path: str, callback, padding: int):
    bin_image = callback(image_path)
    height, width = bin_image.shape

    labels = np.zeros((height, width), dtype=np.int32)

    current_label = 1
    max_area = 0
    largest_bounding_box = None

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if bin_image[y, x] == 255 and labels[y, x] == 0:
                area, min_x, min_y, max_x, max_y = 0, x, y, x, y
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()

                    if labels[cy, cx] != 0:
                        continue

                    labels[cy, cx] = current_label
                    area += 1

                    min_x = min(min_x, cx)
                    min_y = min(min_y, cy)
                    max_x = max(max_x, cx)
                    max_y = max(max_y, cy)

                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if bin_image[ny, nx] == 255 and labels[ny, nx] == 0:
                                stack.append((ny, nx))

                if area > max_area:
                    max_area = area
                    largest_bounding_box = (min_x, min_y, max_x, max_y)

                current_label += 1

    if largest_bounding_box:
        min_x, min_y, max_x, max_y = largest_bounding_box

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(width - 1, max_x + padding)
        max_y = min(height - 1, max_y + padding)

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return min_x, min_y, width, height
    else:
        return None


def draw_bounding_box(image, bounding_box, color=(0, 255, 0), thickness=2):
    x, y, w, h = bounding_box

    image[y:y + thickness, x:x + w] = color
    image[y + h - thickness:y + h, x:x + w] = color

    image[y:y + h, x:x + thickness] = color
    image[y:y + h, x + w - thickness:x + w] = color

    return image


def show_detected_face(image_path: str, callback):
    img = cv2.imread(image_path)
    bounding_box = detect_face(image_path, callback, padding=5)

    if not bounding_box:
        print('Did not detect any faces!')

    image_with_detected_face = draw_bounding_box(img, bounding_box)

    cv2.imshow("Skin Detection", image_with_detected_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_multiple_skin_areas(image_path: str, callback, padding: int, area_threshold=0.25):
    bin_image = callback(image_path)
    height, width = bin_image.shape
    labels = np.zeros((height, width), dtype=np.int32)
    bounding_boxes = []
    current_label = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if bin_image[y, x] == 255 and labels[y, x] == 0:
                area, min_x, min_y, max_x, max_y = 0, x, y, x, y
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if labels[cy, cx] != 0:
                        continue
                    labels[cy, cx] = current_label
                    area += 1
                    min_x = min(min_x, cx)
                    min_y = min(min_y, cy)
                    max_x = max(max_x, cx)
                    max_y = max(max_y, cy)

                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if bin_image[ny, nx] == 255 and labels[ny, nx] == 0:
                                stack.append((ny, nx))


                min_x = max(0, min_x - padding)
                min_y = max(0, min_y - padding)
                max_x = min(width - 1, max_x + padding)
                max_y = min(height - 1, max_y + padding)

                box_width = max_x - min_x + 1
                box_height = max_y - min_y + 1
                bounding_boxes.append((min_x, min_y, box_width, box_height, area))

                current_label += 1

    if not bounding_boxes:
        return []

    max_area = max(bbox[4] for bbox in bounding_boxes)

    significant_boxes = [
        (min_x, min_y, width, height)
        for min_x, min_y, width, height, area in bounding_boxes
        if area >= max_area * area_threshold
    ]

    return significant_boxes


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box

        image[y:y + thickness, x:x + w] = color
        image[y + h - thickness:y + h, x:x + w] = color

        image[y:y + h, x:x + thickness] = color
        image[y:y + h, x + w - thickness:x + w] = color

    return image


def show_multiple_detected_faces(image_path: str, callback, area_threshold):
    img = cv2.imread(image_path)
    bounding_boxes = detect_multiple_skin_areas(image_path, callback, padding=5, area_threshold=area_threshold)

    if not bounding_boxes:
        print('Did not detect any faces!')

    image_with_detected_face = draw_bounding_boxes(img, bounding_boxes)

    cv2.imshow("Multiple Faces - Skin Detection", image_with_detected_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_path = 'E:\\Master\\ComputerVision\\homework3\\skin\\5.jpg' # works well
    # image_path = 'E:\\Master\\ComputerVision\\homework3\\skin\\manwoman.jpg' # returns woman's face
    # image_path = 'E:\\Master\\ComputerVision\\homework3\\Pratheepan_Dataset\\FacePhoto\\0520962400.jpg' # works only for mtw
    image_path = 'E:\\Master\\ComputerVision\\homework3\\skin\\group.jpg'
    image_path_couple = 'E:\\Master\\ComputerVision\\homework3\\skin\\manwoman.jpg'
    image_path_group = 'E:\\Master\\ComputerVision\\homework3\\skin\\4.jpg'

    # show_detected_face(image_path, mo.handle_skin)
    show_detected_face(image_path_couple, mtw.handle_skin)
    # show_detected_face(image_path, mth.handle_skin)

    show_multiple_detected_faces(image_path_couple, mtw.handle_skin, area_threshold=0.5)
    show_multiple_detected_faces(image_path, mtw.handle_skin, area_threshold=0.2)
    show_multiple_detected_faces(image_path_group, mtw.handle_skin, area_threshold=0.2)
