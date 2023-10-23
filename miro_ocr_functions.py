import asyncio
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
        save_path: str
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


# NOT IN USE!!!

# VARS FOR PADDLE OCR MODEL
# ocr_model = PaddleOCR(lang=config.ocr_model_language,
#                       use_angle_cls=True, show_log=False)
# ocr_confidence_threshold = config.ocr_confidence_threshold

# Visualize ocr bounding boxes and the labels provided in ocr_recognized_text_and_boxes_array inside the image with img_path
# (optional function - only when save_image_overlayed_with_ocr_visualization flag is true)
# def get_image_with_overlayed_ocr_bounding_boxes_and_text(img_path: str, ocr_recognized_text_and_boxes_array):
#     img = cv2.imread(img_path)
#     for ocr_recognized_text_and_box in ocr_recognized_text_and_boxes_array:
#         xmin = ocr_recognized_text_and_box['position']['xmin']
#         ymin = ocr_recognized_text_and_box['position']['ymin']
#         xmax = ocr_recognized_text_and_box['position']['xmax']
#         ymax = ocr_recognized_text_and_box['position']['ymax']
#         text = ocr_recognized_text_and_box['text']

#         img = cv2.rectangle(
#             img=img,
#             pt1=(xmin, ymin),
#             pt2=(xmax, ymax),
#             color=(255, 255, 255),
#             thickness=2,
#             lineType=cv2.LINE_AA)

#         img = cv2.putText(
#             img=img,
#             text=text,
#             org=(int(xmin + 5), int(ymin - 5)),
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=0.5,
#             color=(255, 255, 255),
#             thickness=1,
#             lineType=cv2.LINE_AA)
#     return img


# Euclidean distance: Given list of points on 2-D plane and an integer K (origin of the coordinate system (0,0))
# Calculate distance to K to find the closest points to the origin
# √{(x2-x1)2 + (y2-y1)2}
# def euclidean(text_and_boxes_array):
#     xmin = text_and_boxes_array['position']['xmin']
#     ymin = text_and_boxes_array['position']['ymin']
#     coor_origin_x = 0
#     coor_origin_y = 0
#     return ((xmin - coor_origin_x)**2 + (ymin - coor_origin_y)**2)**0.5


# async def get_image_ocr_data(
#     img_path,
#     ocr_confidence_threshold=ocr_confidence_threshold,
# ) -> Tuple[list, Mat]:
#     result = ocr_model.ocr(img_path)

#     # convert all text into array
#     # text_array = [res[1][0] for res in result]
#     # print("\n \n \n")
#     # for text in text_array:
#     #     print(text)

#     # Extracting detected components
#     boxes = [res[0] for res in result]
#     texts = [res[1][0] for res in result]
#     scores = [res[1][1] for res in result]

#     # TODO: FIND OUT IF CONVERTION IS REALLY NEEDED - MESSES UP SAVED FILES
#     # By default if we import image using openCV, the image will be in BGR
#     # But we want to reorder the color channel to RGB for the draw_ocd method
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

#     ocr_recognized_text_and_boxes_array = []

#     # Visualize image and detections
#     for i in range(len(result)):
#         if scores[i] > ocr_confidence_threshold:
#             (xmin, ymin, xmax, ymax) = (
#                 int(boxes[i][0][0]),
#                 int(boxes[i][1][1]),
#                 int(boxes[i][2][0]),
#                 int(boxes[i][3][1]))

#             # position is necessary for the euclidean distance sorting of the textes
#             # (and for the visualization of the bounding boxes if save_image_overlayed_with_ocr_visualization is true)
#             ocr_recognized_text_and_boxes_array.append(
#                 {"position": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
#                  "text": texts[i]}
#             )

#     # Use euclidean distance to sort the words in the correct order from top left to bottom right
#     ocr_recognized_text_and_boxes_array.sort(key=euclidean)

#     return ocr_recognized_text_and_boxes_array


# Remove forbidden printable ASCII characters from file name:
# < > : " / \ | ? * '
def remove_forbidden_ascii_characters(string: str) -> str:
    forbidden_characters = ['<', '>', ':',
                            '"', "\'", '/', '\\', '|', '?', '*']
    for forbidden_character in forbidden_characters:
        if forbidden_character in string:
            string = string.replace(forbidden_character, "")
        # gcloud vision api return "\n" when a line break occures (two paragraphs follow each other)
        # to save the file with the recognized ocr text "\n" must replaced with ""
        if "\n" in string:
            string = string.replace("\n", " ")

    return string


# Create single string out of all ocr data bounding boxes
# def extract_string_out_of_ocr_data(
#     cropped_image: list
# ) -> str:
#     image_ocr_text = ""
#     for ocr_text_and_boxes in cropped_image['ocr_data']:
#         image_ocr_text = image_ocr_text + \
#             ocr_text_and_boxes['text'] + " "
#     return image_ocr_text


def rename_cropped_image_with_ocr_string(cropped_image: list, index: int, image_file_path, timestamped_folder_path: str):
    new_image_file_name = f"{index}_{cropped_image['ocr_recognized_text']}"
    new_image_file_name = remove_forbidden_ascii_characters(
        new_image_file_name)
    new_image_file_path = os.path.join(
        timestamped_folder_path, f"{new_image_file_name}.png")
    os.rename(image_file_path, new_image_file_path)
    return [new_image_file_name, new_image_file_path]


# WAS USED, BUT IS CURRENTLY NOT IN USE
# http://color.lukas-stratmann.com/color-systems/hsv.html
# Get dominant color and assign it to miro sticky note color class
# sticky_note_dominant_rgb_color = get_dominant_color(new_image_file_name_with_ocr_text_in_file_name)
# sticky_note_color_class = get_sticky_note_color_class_from_rgb(sticky_note_dominant_rgb_color)

# resize image down to 1 pixel.
# def get_dominant_color(img_path):
#     img = Image.open(img_path)
#     img = img.convert("RGB")
#     img = img.resize((1, 1), resample=0)
#     dominant_rgb_color = img.getpixel((0, 0))
#     return dominant_rgb_color


# def rgb_to_hsv(r, g, b) -> Tuple[int, int, int]:
#     temp = colorsys.rgb_to_hsv(r, g, b)
#     h = int(temp[0] * 360)
#     s = int(temp[1] * 100)
#     v = round(temp[2] * 100 / 255)
#     return [h, s, v]


# def get_sticky_note_color_class_from_rgb(rgb_color):
#     [h, s, v] = rgb_to_hsv(
#         rgb_color[0],   # r
#         rgb_color[1],   # g
#         rgb_color[2]    # b
#     )

#     sticky_note_color_class = None
#     hue = h

#     if hue > 10 and hue < 60:
#         sticky_note_color_class = "yellow"
#     elif hue > 60 and hue < 180:
#         sticky_note_color_class = "green"
#     elif hue > 180 and hue < 290:
#         sticky_note_color_class = "blue"
#     elif hue > 290 or hue > 0 and hue < 10:
#         sticky_note_color_class = "red"

#     return sticky_note_color_class


def morphology_closing_operation(img):
    # Reading the input image
    # img = cv2.imread(img_path, 0)

    # Taking a matrix of size 5 as the kernel
    size = (5, 5)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    img_closed = cv2.dilate(img, kernel, iterations=1)
    img_closed = cv2.erode(img, kernel, iterations=1)

    return img_closed


def morphology_opening_operation(img):
    # Reading the input image
    # img = cv2.imread(img_path, 0)

    # Taking a matrix of size 5 as the kernel
    size = (5, 5)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    img_opened = cv2.erode(img, kernel, iterations=1)
    img_opened = cv2.dilate(img, kernel, iterations=1)

    return img_opened


# def opencv_script_thresholding(img_path):
#     img = cv2.imread(img_path, 0)
#     # global thresholding
#     ret1, th1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
#     # Otsu's thresholding
#     ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     # Otsu's thresholding after Gaussian filtering
#     blur = cv2.GaussianBlur(img, (5, 5), 0)
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     # plot all the images and their histograms
#     images = [img, 0, th1,
#               img, 0, th2,
#               blur, 0, th3]
#     titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
#               'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
#               'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
#     for i in range(3):
#         plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
#         plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#         plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
#         plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#         plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
#         plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#     plt.show()

# https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/


def opencv_script_thresholding(img_path):
    img = cv2.imread(img_path, 0)

    # Global thresholding
    ret1, global_threshold_img = cv2.threshold(
        img, 100, 255, cv2.THRESH_BINARY)

    # Adaptive thresholding
    adaptive_threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

    # Adaptive thresholding after Gaussian filtering
    adaptive_threshold_gaussian_blur_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                 cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding
    ret4, otus_threshold_img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret5, otus_threshold_gaussian_blur_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, global_threshold_img,
              img, 0, adaptive_threshold_img,
              blur, 0, adaptive_threshold_gaussian_blur_img,
              img, 0, otus_threshold_img,
              blur, 0, otus_threshold_gaussian_blur_img]

    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', 'Adaptive thresholding',
              'Gaussian filtered Image', 'Histogram', 'Adaptive thresholding after Gaussian filtering',
              'Original Noisy Image', 'Histogram', 'Otsus Thresholding',
              'Gaussian filtered Image', 'Histogram', 'Otsus Thresholding after Gaussian filtering']

    for i in range(5):
        plt.subplot(5, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+3), plt.imshow((images[i*3+2]), 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


# # TEST WITH PYTESSERACT
# # !pip install pytesseract
# # install tesseract via tesseract-ocr-w64-setup-v5.2.0.20220712.exe (64 bit) on https://github.com/UB-Mannheim/tesseract/wiki
# # reference path
# # Ctrl+CLick on pytesseract to see all functions AND Ctrl+Click on function to see function definition
# # myconfig = r"--psm 11 --oem 3"
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# sticky_notes_test_images_path = 'C:\_WORK\\GitHub\_data-science\TFODCourse\sticky_notes_test_images'

# img = cv2.imread(os.path.join(
#     sticky_notes_test_images_path, "medium.jpg"))

# height, width, _ = img.shape

# # WARNING: pytesseract library only accept RGB values and openCV is in BGR
# # Therefore, convertion from BGR to RGB is needed
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # image_to_string
# # Returns result of a Tesseract OCR run on provided image to string
# # print(pytesseract.image_to_string(img))

# data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
# amount_of_boxes = len(data['text'])

# for i in range(amount_of_boxes):
#     if float(data['conf'][i]) > 80:
#         (x, y, width, height) = (
#             data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         img = cv2.rectangle(
#             img, (x, y), (x + width, y + height), (0, 255, 0), 2)
#         img = cv2.putText(img, data['text'][i], (x, y + height+20),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.cv2.LINE_AA)

# cv2.imshow('img', img)
# cv2.waitKey(0)
