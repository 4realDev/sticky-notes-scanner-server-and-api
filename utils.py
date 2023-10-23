from datetime import date, datetime
import os
from PIL import Image, ImageDraw
import io
import cv2
from cv2 import Mat
import numpy as np

import config
paths = config.paths


# https://stackoverflow.com/questions/52940369/is-it-possible-to-resize-an-image-by-its-bytes-rather-than-width-and-height
def limit_img_size(img_path, target_filesize):
    img = img_orig = Image.open(img_path)
    aspect = img.size[0] / img.size[1]

    while True:
        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            data = buffer.getvalue()
        filesize = len(data)
        size_deviation = filesize / target_filesize
        # print("size: {}; factor: {:.3f}".format(filesize, size_deviation))

        if size_deviation <= 1:
            # filesize fits
            with open(img_path, "wb") as f:
                f.write(data)
            break
        else:
            # filesize not good enough => adapt width and height
            # use sqrt of deviation since applied both in width and height
            new_width = img.size[0] / size_deviation**0.5
            new_height = new_width / aspect
            # resize from img_orig to not lose quality
            img = img_orig.resize((int(new_width), int(new_height)))

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    return img


def get_timestamp_yyyy_mm_dd_hh_mm_ss() -> str:
    return (str(date.today().year) + '-' +
            str(date.today().month).zfill(2) + '-' +
            str(date.today().day).zfill(2) + '-' +
            str(datetime.now().hour) + '-' +
            str(datetime.now().minute) + '-' +
            str(datetime.now().second))


# Use the given timestamp to create a special folder for the original image,
# the original image with the detected sticky note bounding boxes
# and a crop images for every detected sticky note in the original image
def create_timestamp_folder_and_return_its_path(folder_name: str) -> str:
    recognized_images_timestamp_folder_path: str = os.path.join(
        paths['MIRO_TIMEFRAME_SNAPSHOTS'], folder_name)

    if not os.path.exists(paths['MIRO_TIMEFRAME_SNAPSHOTS']):
        os.mkdir(paths['MIRO_TIMEFRAME_SNAPSHOTS'])

    if not os.path.exists(recognized_images_timestamp_folder_path):
        os.mkdir(recognized_images_timestamp_folder_path)

    return recognized_images_timestamp_folder_path


# Saves the image img with the name img_name inside the folder with the path folder_path
def save_image_in_folder(img: np.ndarray, img_name: str, folder_path: str) -> None:
    img_path = os.path.join(folder_path, img_name)
    result: bool = cv2.imwrite(img_path, img)
    if result == True:
        print(f"\nFile {img_name} saved successfully.\n")
    else:
        print(f"\nError in saving file {img_name}.\n")


def draw_boxes(img: Mat, texts_image_data, color="yellow", fillcolor=None):
    """Draw a border around the image using the hints in the vector list."""
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image, "RGB")

    for data in texts_image_data:
        draw.polygon(
            xy=[
                data["boxes"]["topLeft"]["x"],
                data["boxes"]["topLeft"]["y"],
                data["boxes"]["topRight"]["x"],
                data["boxes"]["topRight"]["y"],
                data["boxes"]["bottomRight"]["x"],
                data["boxes"]["bottomRight"]["y"],
                data["boxes"]["bottomLeft"]["x"],
                data["boxes"]["bottomLeft"]["y"],
            ],
            fill=fillcolor,
            outline=color,
            width=10
        )

    image.load()
    return np.array(image)


def draw_rectangles(img: Mat, detections, color="None", fillcolor="white", width=1):
    """Draw a border around the image using the hints in the vector list."""
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image, "RGB")

    for detection in detections:
        draw.rectangle(
            xy=[
                (detection["xmin"], detection["ymin"]),
                (detection["xmax"], detection["ymax"]),
            ],
            fill=fillcolor,
            outline=color,
            width=width
        )
    image.load()
    return np.array(image)
