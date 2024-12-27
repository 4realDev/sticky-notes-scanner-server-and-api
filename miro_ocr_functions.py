import asyncio
import tempfile
from cv2 import Mat
import config
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from enum import Enum
from PIL import Image, ImageDraw


# Imports the Google Cloud client library
from google.cloud import vision
import io

import nest_asyncio

from utils import save_image_in_folder

nest_asyncio.apply()

paths = config.paths


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


# https://cloud.google.com/java/docs/reference/google-cloud-vision/latest/com.google.cloud.vision.v1.TextAnnotation.DetectedBreak.BreakType
class TextAnnotationDetectedBreakType(Enum):
    SPACE = 1
    EOL_SURE_SPACE = 3
    LINE_BREAK = 5

# Saves a cropped image for every detected sticky note bounding box inside the original image
# with the detection bounding-box coordinates from img_detection_data_list
# and return a list with the temporary name of the image, the position of the bounding box and the detected color
def crop_image_to_bounding_boxes_and_save_them(
        img: np.ndarray,
        img_detection_data_list,
        save_path: str,
        DEBUG: bool
) -> list:

    cropped_images_data = []

    for index, img_detection_data in enumerate(img_detection_data_list):
        ymin = img_detection_data['ymin']
        ymax = img_detection_data['ymax']
        xmin = img_detection_data['xmin']
        xmax = img_detection_data['xmax']
        color = img_detection_data['color']

        cropped_img_according_to_its_bounding_box = img[ymin:ymax, xmin:xmax]
        cropped_img_according_to_its_bounding_box_name = f"cropped_image_{index}.png"
        
        # TODO: Tempfile logic here - Adjust delete to False again
        with tempfile.NamedTemporaryFile(suffix=".jpg", mode="wb", delete=False) as temp_jpg:
            # TODO: Maybe find a better way without saving the image twice (only because of gcloud API)
            success, encoded_img = cv2.imencode(".jpg", cropped_img_according_to_its_bounding_box)
            temp_jpg.write(encoded_img.tobytes())
            cropped_img_according_to_its_bounding_box_full_path = temp_jpg.name
        
            if DEBUG:
                cropped_img_according_to_its_bounding_box_full_path = os.path.join(
                    save_path, cropped_img_according_to_its_bounding_box_name)

                # Save cropped images of the detected objects inside the bounding boxes in dedicated backup folder
                # necessary for OCR, because OCR does only work with path of existing image file
                save_image_in_folder(
                    cropped_img_according_to_its_bounding_box,
                    cropped_img_according_to_its_bounding_box_name,
                    save_path
                )

                print("\n")

            cropped_images_data.append(
                {
                    "position": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                    "width": xmax - xmin,
                    "height": ymax - ymin,
                    "color": color,
                    "name": cropped_img_according_to_its_bounding_box_name,
                    "ocr_recognized_text": "",
                    "path": cropped_img_according_to_its_bounding_box_full_path
                })
        
    return cropped_images_data


async def get_ocr_detection_text_and_bounding_boxes(
    img_path: str,
):
    text_ocr_detections = await asyncio.create_task(
        detect_text_with_gcloud_vision_api(img_path)
    )

    # returns the bounding boxes of the text blocks inside the text_ocr_detections
    # [
    #     "topLeft": {"x": int, "y": int},
    #     "topRight": {"x": int, "y": int},
    #     "bottomLeft": {"x": int, "y": int},
    #     "bottomRight": {"x": int, "y": int},
    # ]
    text_box_and_text_array = get_detected_text_block_bounding_boxes(
        text_ocr_detections)

    return text_box_and_text_array


# https://stackoverflow.com/questions/51972479/get-lines-and-paragraphs-not-symbols-from-google-vision-api-ocr-on-pdf
def get_detected_text_block_bounding_boxes(detection):
    bounds = []
    lines = []
    breaks = vision.TextAnnotation.DetectedBreak.BreakType

    for page in detection.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                block_text = ""
                line = ""
                for word in paragraph.words:
                    for symbol in word.symbols:
                        line += symbol.text
                        if symbol.property.detected_break.type == breaks.SPACE:
                            line += ' '
                        if symbol.property.detected_break.type == breaks.EOL_SURE_SPACE:
                            line += ' '
                            lines.append(line)
                            block_text += line
                            line = ''
                        if symbol.property.detected_break.type == breaks.LINE_BREAK:
                            lines.append(line)
                            block_text += line
                            line = ''

            block_bounding_box = {
                "topLeft": {"x": block.bounding_box.vertices[0].x, "y": block.bounding_box.vertices[0].y},
                "topRight": {"x": block.bounding_box.vertices[1].x, "y": block.bounding_box.vertices[1].y},
                "bottomLeft": {"x": block.bounding_box.vertices[3].x, "y": block.bounding_box.vertices[3].y},
                "bottomRight": {"x": block.bounding_box.vertices[2].x, "y": block.bounding_box.vertices[2].y},
            }
            
            # remove black circles from voting dot detection, if there are any
            block_text = block_text.replace("●", "")


            bounds.append(
                {"boxes": block_bounding_box, "text": block_text})

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


# https://cloud.google.com/vision/docs/detect-labels-image-client-libraries?cloudshell=false#local-shell
# pip install --upgrade google-cloud-vision
# afterwards you must run: pip install --upgrade protobuf==3.20.0 - otherwise there will be problems with tensorflow

# The Vision API can detect and extract text from images
# DOCUMENT_TEXT_DETECTION extracts text from an image (or file);
# the response is optimized for dense text and documents.
# The JSON includes page, block, paragraph, word, and break information.
# One specific use of DOCUMENT_TEXT_DETECTION is to detect handwriting in an image.
# in this example, we use the feature detection to detect text in a local image file
# for it to work we need to send the contents of the image file as base64 encoded string in the body of the request
# (https://cloud.google.com/vision/docs/base64)

# Specify the language
# one or more languageHints can specify the language of any text in image
# however, empty value yields often the best results, because omitting a value enables automatic language detection
# for languages based on Latin alphabet, setting languageHints is not needed
# path: path to the image file
async def detect_text_with_gcloud_vision_api(path: str) -> str:
    """Detects document features in an image."""

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs document text detection on the image file
    response = client.document_text_detection(
        image=image, image_context={"language_hints": ["de", "en"]})

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return response.full_text_annotation