IMPORTANT: Currently only working with Python 3.10.7

1.
py -m venv env
.\env\Scripts\activate

2.
pip install PyYAML==5.3.1

3.
git clone https://github.com/tensorflow/models.git

copy <YOUR_PATH>\models\research\object_detection\packages\tf2\setup.py <YOUR_PATH>\models\research

Filled in:
copy C:\_WORK\GitHub\toolbox-scanning-api\models\research\object_detection\packages\tf2\setup.py C:\_WORK\GitHub\toolbox-scanning-api\models\research

cd models\research
python -m pip install --use-feature=2020-resolver .

4.
pip install --upgrade protobuf==3.20.0

6.
protoc object_detection/protos/*.proto --python_out=.



7.
cd ..\..
pip install -r requirements.txt
(
flask
flask_cors
flask[async]
aiohttp
nest_asyncio
google-cloud-vision
protobuf==3.20.0
)

8.
python -m flask --app .\app.py run
OR
python -m flask --app flaskr run
