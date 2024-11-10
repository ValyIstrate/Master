import method_one as mo
import method_two as mtw
import method_three as mth
import os
import cv2
import numpy as np
from datetime import datetime


class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def compute_accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        if total == 0:
            return 0.0
        return (self.tp + self.tn) / total

    def add_values(self, values):
        self.tp += values[0]
        self.tn += values[1]
        self.fp += values[2]
        self.fn += values[3]

    def write_accuracy_to_file(self, filename, method_name):
        accuracy = self.compute_accuracy()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filename, 'w') as file:
            file.write(f"Executed at date {current_time} for method {method_name}. \n")
            file.write(f"Result: {accuracy:.4f}\n")


def validate_for_method(obtained_image, gt_img):
    if obtained_image.shape != gt_img.shape:
        raise ValueError("Images have different sizes, cannot compare pixel values.")

    true_positives = np.sum((obtained_image == 255) & (gt_img == 255))
    true_negatives = np.sum((obtained_image == 0) & (gt_img == 0))
    false_positives = np.sum((obtained_image == 0) & (gt_img == 255))
    false_negatives = np.sum((obtained_image == 255) & (gt_img == 0))

    return true_positives, true_negatives, false_positives, false_negatives


def validate_methods(dataset_path: str, ground_truth_path: str):
    gt_face_folder = os.path.join(ground_truth_path, 'GroundT_FacePhoto')
    dataset_face_folder = os.path.join(dataset_path, 'FacePhoto')
    gt_family_folder = os.path.join(ground_truth_path, 'GroundT_FamilyPhoto')
    dataset_family_folder = os.path.join(dataset_path, 'FamilyPhoto')

    mo_confusion_matrix = ConfusionMatrix()
    mtw_confusion_matrix = ConfusionMatrix()
    mth_confusion_matrix = ConfusionMatrix()

    face_results = get_confusion_matrix_parameters_for_dataset(dataset_face_folder, gt_face_folder)
    family_results = get_confusion_matrix_parameters_for_dataset(dataset_family_folder, gt_family_folder)

    mo_confusion_matrix.add_values(face_results[0])
    mo_confusion_matrix.add_values(family_results[0])

    mtw_confusion_matrix.add_values(face_results[1])
    mtw_confusion_matrix.add_values(family_results[1])

    mth_confusion_matrix.add_values(face_results[2])
    mth_confusion_matrix.add_values(family_results[2])

    mo_confusion_matrix.write_accuracy_to_file("results/Method_One.txt", "Method One")
    mtw_confusion_matrix.write_accuracy_to_file("results/Method_Two.txt", "Method Two")
    mth_confusion_matrix.write_accuracy_to_file("results/Method_Three.txt", "Method Two")


def get_confusion_matrix_parameters_for_dataset(dataset, ground_truth):
    mo_tp, mo_tn, mo_fp, mo_fn = 0, 0, 0, 0
    mtw_tp, mtw_tn, mtw_fp, mtw_fn = 0, 0, 0, 0
    mth_tp, mth_tn, mth_fp, mth_fn = 0, 0, 0, 0

    for img_name in os.listdir(ground_truth):
        gt_img = cv2.imread(os.path.join(ground_truth, img_name), cv2.IMREAD_GRAYSCALE)

        data_img_name = img_name.replace('.png', '.jpg') if img_name.endswith('.png') else img_name
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)

        mo_values = validate_for_method(mo.handle_skin(os.path.join(dataset, data_img_name)), gt_img)
        mtw_values = validate_for_method(mtw.handle_skin(os.path.join(dataset, data_img_name)), gt_img)
        mth_values = validate_for_method(mth.handle_skin(os.path.join(dataset, data_img_name)), gt_img)

        mo_tp += mo_values[0]
        mo_tn += mo_values[1]
        mo_fp += mo_values[2]
        mo_fn += mo_values[3]

        mtw_tp += mtw_values[0]
        mtw_tn += mtw_values[1]
        mtw_fp += mtw_values[2]
        mtw_fn += mtw_values[3]

        mth_tp += mth_values[0]
        mth_tn += mth_values[1]
        mth_fp += mth_values[2]
        mth_fn += mth_values[3]

    return (mo_tp, mo_tn, mo_fp, mo_fn), (mtw_tp, mtw_tn, mtw_fp, mtw_fn), (mth_tp, mth_tn, mth_fp, mth_fn)


if __name__ == '__main__':
    dataset_path = r'E:\Master\ComputerVision\homework3\Pratheepan_Dataset'
    ground_truth_set_path = r'E:\Master\ComputerVision\homework3\Ground_Truth'
    validate_methods(dataset_path, ground_truth_set_path)
