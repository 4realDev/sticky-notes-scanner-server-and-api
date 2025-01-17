## Sticky Note Scanner API

## I. SETUP AND INSTALLATION

IMPORTANT: 
- Currently only tested with `Python 3.10.7`  
- Run all commands in Windows PowerShell

### 1.  Create Virtual Environment
Advantages:  
    - Isolates all Python and library dependencies needed for TFOD model  
    - Ensures clean working environment  
    - Does not conflict with all other already installed libraries and dependencies on your machine  
    -> no dependency conflicts  
    -> Virtual environment has it’s own isolated packages under Lib/site-packages  

```
py -m venv env
```

### 2.  Start Virtual Environment with windows batch script in `Scripts/activate`
(env) is visible and pip list only shows isolated packages (Lib/site-packages) in your environment  

```
.\env\Scripts\activate
```

### 3.  Update pip Libraries and Dependencies
Ensure that we have latest resolvers and upgraded pip install app  
In virtual environment run:  

```
python -m pip install --upgrade pip
```

### 4.  Install `PyYAML` -> necessary for TensorFlow Object Detection

```
pip install PyYAML==5.3.1
```

### 5.  Clone Tensorflow Object Detection API / TensorFlow Model Garden

```
git clone https://github.com/tensorflow/models.git
```

### 6.  Copy the `setup.py` inside the `models/research` folder 

```
copy <YOUR_PATH>\models\research\object_detection\packages\tf2\setup.py <YOUR_PATH>\models\research
```

### 7.  Run setup.py to install all necessary libraries for TensorFlow Object Detection

```
cd models\research
```

```
python -m pip install --use-feature=2020-resolver .
``` 

### 8.  Install Protoc and run protoc and all `.proto` files inside `models/research/object_detection/protos` folder
Protoc "used to compile .proto files, which contain service and message definitions"  

```
pip install --upgrade protobuf==3.20.0
```

```
protoc object_detection/protos/*.proto --python_out=.
```

### 9.  Install all required libraries for the API

```
cd ..\..
```

```
pip install -r requirements.txt
```   

Those are: flask, flask_cors, flask[async], opencv-python, aiohttp, nest_asyncio, google-cloud-vision
<br>
<br>

## II. START API

### 1. Start the Virtual Environment

```
.\env\Scripts\activate
```

### 2.  Start the Server (localhost:5000)

```
python -m flask --app flaskr run
```

## III. Returned Data
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
- the original image with the timestamp (o.i.)
- the object detection of the sticky notes inside the o.i.
- the optical character recognition of the sticky note texts inside the o.i.
- all sticky notes cropped out of the o.i. as single images
