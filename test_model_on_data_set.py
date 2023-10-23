# (in scripts) python own-scripts\preprocessing\run_image_auto_labelling.py -imgp C:\Users\vbraz\Desktop\ML_Images_21-02-23 -mth 0.9
# IMPORTANT: running this script via console, make sure you are in scripts direcory!

# This library can be used to create image annotation XML files in the PASCAL VOC file format.
import argparse
import os
import numpy as np

import config
import cv2

from miro_tfod_functions import (
    get_detections_from_img,
    get_image_with_overlayed_labeled_bounding_boxes,
    load_latest_checkpoint_of_custom_object_detection_model)

from utils import \
    get_timestamp_yyyy_mm_dd_hh_mm_ss, \
    save_image_in_folder

files = config.files
paths = config.paths
min_score_thresh = config.min_score_thresh
bounding_box_and_label_line_thickness = config.bounding_box_and_label_line_thickness

data_set_default_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "test_data_set")


parser = argparse.ArgumentParser(
    description="Script to test tensorflow custom object detection models on a test data set stored in the test_data_set-folder in the project.")
parser.add_argument("-d",
                    "--data_set_path",
                    help="Path to the folder where the test images are stored.",
                    metavar='data_set_path',
                    type=str,
                    default=data_set_default_path,
                    required=False)
parser.add_argument("-mth",
                    "--min_threshold",
                    metavar='min_threshold',
                    help="Minimal Thredhold for the TensorFlow Object Detection inside the images.",
                    type=float,
                    default=min_score_thresh,
                    required=False)


args = parser.parse_args()

debug_image_count = 0


def test_model_on_data_set(data_set_path: str, min_score_thresh: float):
    load_latest_checkpoint_of_custom_object_detection_model()

    if not os.path.exists(data_set_path):
        os.mkdir(data_set_path)

    if len(os.listdir(data_set_path)) == 0:
        print("Please add images in your 'test_data_set' folder, to test your selected model with those images")
        return

    test_results_folder_path: str = os.path.join(
        data_set_path,
        f"{config.CUSTOM_MODEL_NAME_SUFFIX}_{str(config.num_steps)}_{get_timestamp_yyyy_mm_dd_hh_mm_ss()}"
    )

    if not os.path.exists(test_results_folder_path):
        os.mkdir(test_results_folder_path)

    # get the path or directory
    for image_name in os.listdir(data_set_path):

        # check if the image ends with png or jpg or jpeg
        image_name_with_lowercase_ending = image_name.lower()
        if image_name_with_lowercase_ending.endswith(".png") or \
                image_name_with_lowercase_ending.endswith(".jpg") or \
                image_name_with_lowercase_ending.endswith(".jpeg"):

            image_path = os.path.join(data_set_path, image_name)
            img = cv2.imread(image_path)
            img_detections = get_detections_from_img(img)

            img_with_visualized_detections = get_image_with_overlayed_labeled_bounding_boxes(
                image=img,
                detections=img_detections,
                min_score_thresh=min_score_thresh
            )

            global debug_image_count
            debug_image_count = debug_image_count + 1
            new_image_name: str = f"{debug_image_count}_{config.CUSTOM_MODEL_NAME_SUFFIX}_{str(config.num_steps)}_{image_name}"

            save_image_in_folder(img_with_visualized_detections,
                                 new_image_name, test_results_folder_path)


def test_on_image(img: np.array, min_score_thresh: float = min_score_thresh):
    global data_set_default_path

    load_latest_checkpoint_of_custom_object_detection_model()

    if not os.path.exists(data_set_default_path):
        os.mkdir(data_set_default_path)

    test_results_folder_path: str = os.path.join(
        data_set_default_path,
        f"{config.CUSTOM_MODEL_NAME_SUFFIX}_{str(config.num_steps)}_{get_timestamp_yyyy_mm_dd_hh_mm_ss()}"
    )

    if not os.path.exists(test_results_folder_path):
        os.mkdir(test_results_folder_path)

    image_name = f"snapshot_{get_timestamp_yyyy_mm_dd_hh_mm_ss()}.png"
    # image_path = os.path.join(data_set_default_path, image_name)
    # img = cv2.imread(image_path)
    img_detections = get_detections_from_img(img)

    img_with_visualized_detections = get_image_with_overlayed_labeled_bounding_boxes(
        image=img,
        detections=img_detections,
        min_score_thresh=min_score_thresh
    )

    save_image_in_folder(img_with_visualized_detections,
                         image_name, test_results_folder_path)


def main():
    load_latest_checkpoint_of_custom_object_detection_model()
    test_model_on_data_set(
        args.data_set_path,
        args.min_threshold,
    )


if __name__ == '__main__':
    main()
