1.
py -m venv env
.\env\Scripts\activate

2.
pip install PyYAML==5.3.1

3.
cd model\research
protoc object_detection/protos/*.proto --python_out=.

?.
pip install --upgrade protobuf==3.20.0

4.
git clone https://github.com/tensorflow/models.git
copy <YOUR_PATH>\models\research\object_detection\packages\tf2 <YOUR_PATH>\models\research
python -m pip install --use-feature=2020-resolver .

5.
cd model\research
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

6.
python -m flask --app .\app.py run