from cv2 import Mat
import numpy as np
import os
import cv2
from genericpath import exists
import asyncio
import nest_asyncio
import tempfile

import config

from miro_tfod_functions import \
    get_image_with_overlayed_labeled_bounding_boxes, \
    get_detections_from_img, \
    get_bounding_boxes_above_min_score_thresh

from miro_ocr_functions import \
    get_ocr_detection_text_and_bounding_boxes, \
    detect_text_with_gcloud_vision_api, \
    crop_image_to_bounding_boxes_and_save_them

from utils import \
    save_image_in_folder, \
    draw_boxes, \
    draw_rectangles


nest_asyncio.apply()

min_score_thresh = config.min_score_thresh


async def get_recognized_sticky_notes_data(
    img_file_path: str,
    timestamped_folder_path: str,
    DEBUG=False
) -> list:

    img = cv2.imread(img_file_path)

    img_detections = get_detections_from_img(img)

    img_detection_data_list = get_bounding_boxes_above_min_score_thresh(
        detections=img_detections,
        imgHeight=img.shape[0],
        imgWidth=img.shape[1],
        min_score_thresh=min_score_thresh
    )

    cropped_bounding_boxes_images_data = crop_image_to_bounding_boxes_and_save_them(
        img,
        img_detection_data_list,
        timestamped_folder_path,
        DEBUG
    )

    print("\n")

    # 1. Get all images in the timestamped_folder_path
    # 2. For each cropped image of the sticky notes inside the timestamped_folder_path: Gets the ocr data array
    # 3. Loop through the image_ocr_data_array, take every ocr bounding box and
    #    create one single string out of all ocr bounding boxes texts for each cropped image of the sticky notes
    # 4. Change the name of the cropped image to the new created single string of the image's ocr text
    # 5. Get the dominant color of every cropped image and its detected sticky note
    #    and assign it to a color class for sticky notes, which miro can understand (with the help of HSV color model)
    # 6. Override cropped_image_data with new data from text and color recognition

    # For each cropped image of the sticky notes inside the timestamped_folder_path
    for index, cropped_image in enumerate(cropped_bounding_boxes_images_data):

        # Get (saved) cropped image of sticky note and check if it is a existing file
        # cropped_image_file_path = os.path.join(
        #     timestamped_folder_path, cropped_image['name'])

        if os.path.isfile(cropped_image["path"]):
            cropped_image_ocr_detections = await asyncio.create_task(
                detect_text_with_gcloud_vision_api(cropped_image["path"])
            )

            cropped_image['ocr_recognized_text'] = cropped_image_ocr_detections.text

    print("\n")

    if DEBUG:
        # Save the original image with the detection bounding boxes of the sticky notes
        # inside the dedicated backup folder (for manual detection control)
        # only necessary for saving the image with the detections inside the backup folder
        img_with_overlayed_labeled_detection_bounding_boxes = get_image_with_overlayed_labeled_bounding_boxes(
            img,
            img_detections,
        )

        save_image_in_folder(
            img_with_overlayed_labeled_detection_bounding_boxes,
            f"_debug_sticky_notes_scan.png",
            timestamped_folder_path,
        )

        # TODO: Most likely remove after masterthesis because it consumes API resources and is not necessary for the further logic
        # Save the original image with the detection bounding boxes of the sticky notes text
        # inside the dedicated backup folder (for manual detection control)
        text_box_and_text_array = await asyncio.create_task(
            get_ocr_detection_text_and_bounding_boxes(img_file_path)
        )

        img_with_all_text_bounding_boxes = draw_boxes(
            img, text_box_and_text_array)

        save_image_in_folder(
            img_with_all_text_bounding_boxes,
            f"_debug_all_texts_scan.png",
            timestamped_folder_path,
        )

    return cropped_bounding_boxes_images_data


async def get_recognized_text_data(img: Mat, dest_folder_path: str, sticky_notes_data, DEBUG=False):
    with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=True) as temp_jpg:
        # TODO: Maybe find a better way without saving the image twice (only because of gcloud API)
        img_text_bounding_boxes_name = temp_jpg.name
        img_text_bounding_boxes_path = temp_jpg.name
        
        sticky_notes_data_detections = (sticky_note['position'] for sticky_note in sticky_notes_data)

        img_with_whiteout_stickies = draw_rectangles(
            img, sticky_notes_data_detections, (255, 255, 0), (255, 255, 0))

        temp_jpg.write(img_with_whiteout_stickies.tobytes())

        
        if DEBUG:
            img_text_bounding_boxes_name = "_debug_non_sticky_note_texts_scan.png"
            img_text_bounding_boxes_path = os.path.join(
                dest_folder_path, img_text_bounding_boxes_name)

            save_image_in_folder(
                img_with_whiteout_stickies,
                img_text_bounding_boxes_name,
                dest_folder_path,
            )
    
        texts_bounding_boxes_images_data = await asyncio.create_task(
            get_ocr_detection_text_and_bounding_boxes(
                img_text_bounding_boxes_path
            )
        )
        
        if DEBUG:
            img_with_whiteout_stickies = draw_boxes(
                img_with_whiteout_stickies, texts_bounding_boxes_images_data)

            save_image_in_folder(
                img_with_whiteout_stickies,
                img_text_bounding_boxes_name,
                dest_folder_path,
            )

        if DEBUG == False:
            os.remove(img_text_bounding_boxes_path)

    return texts_bounding_boxes_images_data
