# Sticky Note Scanner - API

**IMPORTANT:** 
>- Currently only tested with `Python 3.10.7`  
>- Run all commands in Windows PowerShell

## I. SETUP AND INSTALLATION
### 1.  Create Virtual Environment
Clone Repository:

```
git clone https://github.com/4realDev/sticky-notes-scanner-server-and-api.git
```

Navigate into root directory:

```
cd sticky-notes-scanner-server-and-api
```

Create Virtual Environment.

```
py -m venv venv
```

Advantages:  
- Isolates all Python and library dependencies needed for TFOD model  
- Ensures clean working environment  
- Does not conflict with all other already installed libraries and dependencies on your machine  
-> no dependency conflicts  
-> Virtual environment has it’s own isolated packages under Lib/site-packages  

### 2.  Start Virtual Environment with windows batch script in `Scripts/activate`
(env) is visible and pip list only shows isolated packages (Lib/site-packages) in your environment  

```
.\venv\Scripts\activate
```

### 3.  Update pip Libraries and Dependencies
Ensure that we have latest resolvers and upgraded pip install app  
In virtual environment run:  

```
python -m pip install --upgrade pip
```

### 4.  Install all required libraries for the API

```
pip install -r requirements.txt
```   

Those are: flask, flask_cors, flask[async], opencv-python, aiohttp, nest_asyncio, google-cloud-vision

### 5.  Clone Tensorflow Object Detection API / TensorFlow Model Garden

```
git clone https://github.com/tensorflow/models.git
```

### 6.  Copy the `setup.py` inside the `models/research` folder 

```
copy .\models\research\object_detection\packages\tf2\setup.py .\models\research
```

### 7.  Run setup.py to install all necessary libraries for TensorFlow Object Detection

```
cd models\research
```

```
python -m pip install .
``` 

### 8.  Install Protoc and run protoc and all `.proto` files inside `models/research/object_detection/protos` folder
Protoc "used to compile .proto files, which contain service and message definitions"  

```
protoc object_detection/protos/*.proto --python_out=.
```

```
pip install --upgrade protobuf==3.20.3
```

### 9. Setup Google Cloud Vision API for Optical Character Recognition
***Follow the instructions inside:*** [README_how_to_register_with_google_cloud_api_ocr](https://github.com/4realDev/sticky-notes-scanner-server-and-api/blob/main/README_how_to_register_with_google_cloud_api_ocr.pdf)

or

Google Setup Documentation ["Cloud Vision setup and cleanup"](https://cloud.google.com/vision/docs/setup)\
Google Usage Documentation ["Using the Vision API"](https://gcloud.readthedocs.io/en/latest/vision-usage.html)

## II. START API

### 1. Start the Virtual Environment

```
.\venv\Scripts\activate
```

### 2.  Start the Server (localhost:5000)

```
python -m flask --app flaskr run
```

### 3. Troubleshoot
Sometimes there is a problem with the `protobuf` version.
If the following problem occures, run those commands:

`ImportError: cannot import name 'string_int_label_map_pb2' from 'object_detection.protos' (C:\GitHub\toolbox-scanning-api\venv\lib\site-packages\object_detection\protos\__init__.py)`

```
cd .\models\research
```

```
python -m pip install .
```

```
pip uninstall protobuf
```

```
pip install --upgrade protobuf==3.20.3
```

```
cd ..\..
```

## III. API Returned Data Type
```
{
    img_data: {
        width: number;
        height: number;
    },
    sticky_note_data: Array<{
        color: string;
        height: number;
        name: string;
        ocr_recognized_text: string;
        path: string;
        position: {
            ymin: number;
            xmin: number;
            ymax: number;
            xmax: number;
        };
        width: number;
    }>,
}
```

***Image of an HTTP example request and return in Insomnia:***
![image](https://github.com/user-attachments/assets/305bbdb3-3275-41a1-b631-7802df6f406d)

## IV. API Request
```
http://localhost:5000/scan-sticky-notes
```

with optional parameters:
```
http://localhost:5000/scan-sticky-notes?debug=true
```

`debug: boolean = False`  

If the `debug` flag is set to `True`, the api will create a folder named `miro-timeframe-snapshots` in which all the important data will be saved as images.
For every scan there will be a subfolder inside `miro-timeframe-snapshots` named after the timestamp -> `miro-timeframe-snapshots/[timestamp]`
Inside the subfolder of each scan the following images will be saved:
- the original image with the timestamp
- the cropped sticky note png images named `cropped_image_x.png`
- the google cloud vision api OCR text detection named `_debug_all_texts_scan.png`
- the tensorflow custom sticky note detection named `_debug_sticky_notes_scan.png`

![Screenshot 2025-02-17 144708](https://github.com/user-attachments/assets/cf1a94b2-d1ce-4f82-b585-698213c5d3bb)

<br/>
<br/>
<br/>
<br/>


---

## Sticky Note Scanner
### Preview
***Showreal Video:*** https://vimeo.com/803506423

<br/>

***Image of a Miro Whiteboard Scan/Digitalisation with:***
![2_Eval_Homeoffice_Phase_4](https://github.com/user-attachments/assets/a6f2cbac-0a08-4ea7-8645-9c820c1db46d)
- all digitalised sticky notes on the left (`text, color, position, size`),
- the cropped out original sticky notes in the middle (`cropped-image, position, size`),
- and the original scan on the right:

<br/>

***Debug Image of the Optical Character Recognition with the Google Cloud Vision API and its OCR Functionality:***
![_debug_all_texts_scan](https://github.com/user-attachments/assets/637f01e6-1113-4540-b888-a30056a5b3e8)

<br/>

***Debug Image of the Sticky Note Object Detection with the TensorFlow Custom Object Detection Model:***
![_debug_sticky_notes_scan](https://github.com/user-attachments/assets/3ba7977c-4dbc-43a7-8e5a-1551cdf8c200)

<br/>
