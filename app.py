# run server locally
# .\venv\Scripts\activate
# python -m flask --app .\app.py run

import os
from flask import Flask, request
from flask_cors import CORS
import numpy as np
import os
import cv2
from genericpath import exists
import aiohttp
import asyncio
import nest_asyncio

from miro_sticky_notes_sync import get_recognized_sticky_notes_data, get_recognized_text_data

from utils import \
    create_timestamp_folder_and_return_its_path, \
    get_timestamp_yyyy_mm_dd_hh_mm_ss, \
    limit_img_size, \
    save_image_in_folder

nest_asyncio.apply()

app = Flask("__name__")
# to solve cross origin issue
cors = CORS(app)


@app.route("/scan-sticky-notes", methods=["POST"])
async def scan_sticky_notes():

    # 1. GET UPLOADED IMAGE
    # check if the post request has the file part
    if 'image' not in request.files:
        print("Missing File")
        return "Please add a file to the body of the post request."

    img = request.files["image"]
    numpy_array_img = cv2.imdecode(
        np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
    img_filename_with_extension = img.filename
    img_extension = os.path.splitext(img_filename_with_extension)[1]

    if (img_extension == ".jpg") or (img_extension == ".png"):
        print("File accepted")
    else:
        return f"Wrong file extension {img_extension}. Only .jpg and .png are accepted."

    print("\n")

    # 2. GET FLAG IF NON STICKY NOTE TEXT SHOULD BE SCANNED AS WELL
    if "scan_whiteboard_text" in request.args:
        scan_whiteboard_text = True
    else:
        scan_whiteboard_text = False
    print(f"scan_whiteboard_text flag is set to: {scan_whiteboard_text}")

    if "scan_voting_dots" in request.args:
        scan_voting_dots = True
    else:
        scan_voting_dots = False
    print(
        f"scan_voting_dots flag is set to: {scan_voting_dots}")

    if "debug" in request.args:
        DEBUG = True
    else:
        DEBUG = False
    print(f"debug flag is set to: {DEBUG}")

    print("\n")

    # 3. CREATE TIMESTAMP FOLDER STRUCUTRE
    timestamp = get_timestamp_yyyy_mm_dd_hh_mm_ss()
    timestamped_folder_name = timestamp
    timestamped_folder_path = create_timestamp_folder_and_return_its_path(
        timestamped_folder_name)

    # 4. SAVE IMAGE IN TIMESTAMP FOLDER
    # TODO: Make this inside cloud
    img_file_path = f"{os.path.join(timestamped_folder_path, img_filename_with_extension)}"

    save_image_in_folder(
        numpy_array_img,
        img_filename_with_extension,
        timestamped_folder_path
    )

    [width, height, numpy_array_img] = limit_img_size(
        img_file_path,
        4500000,   # bytes
    )

    # 5. START SCANNING PROCESS
    async with aiohttp.ClientSession() as session:

        # 6. SCAN STICKY NOTE OBJECTS AND THEIR OCR RECOGNIZED TEXT
        # Use TFOD, Google Cloud Vision API and OpenCV to get the recognized sticky notes data
        # in the given image from the passed img_file_path with the following information:
        # bounding-box position         | sticky_note_data['position'] -> {"xmin": int, "xmax": int, "ymin": int, "ymax": int}
        # image width                   | sticky_note_data['width'] -> int
        # image height                  | sticky_note_data['height'] -> int
        # recognized color              | sticky_note_data['color'] -> str
        # file name of saved image      | sticky_note_data['name'] -> str
        # detected ocr-text             | sticky_note_data['ocr_recognized_text'] -> str
        # image as matrix array         | sticky_note_data['image'] -> array
        # path where image was saved    | sticky_note_data['path'] -> str
        sticky_notes_data = await asyncio.create_task(
            get_recognized_sticky_notes_data(
                img_file_path,
                timestamped_folder_path,
                scan_voting_dots,
                DEBUG
            )
        )

        # 7. SCAN TEXT ON WHITEBOARD
        text_data = []
        if scan_whiteboard_text:
            # override all detections of the sticky notes with white polygones
            # to be able to detect only the remaining handwritten text
            text_data = await asyncio.create_task(
                get_recognized_text_data(
                    numpy_array_img,
                    timestamped_folder_path,
                    sticky_notes_data,
                    DEBUG
                )
            )

            return {"img_data": {"width": width, "height": height}, "sticky_note_data": sticky_notes_data, "text_data": text_data}

        else:
            return {"img_data": {"width": width, "height": height}, "sticky_note_data": sticky_notes_data}
