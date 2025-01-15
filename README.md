## Sticky Note Scanner API

### Setup Guide

IMPORTANT: Currently only tested with `Python 3.10.7`

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

#### 8.  Install Protoc and run protoc and all `.proto` files inside `models/research/object_detection/protos` folder
Protoc "used to compile .proto files, which contain service and message definitions"  

```
pip install --upgrade protobuf==3.20.0
```

```
protoc object_detection/protos/*.proto --python_out=.
```

#### 9.  Install all required libraries for the API

```
cd ..\..
```

```
pip install -r requirements.txt
```   

Those are: flask, flask_cors, flask[async], opencv-python, aiohttp, nest_asyncio, google-cloud-vision

#### 10. Start API Server

```
python -m flask --app flaskr run
```
