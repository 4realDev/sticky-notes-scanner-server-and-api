1.
py -m venv env
.\env\Scripts\activate

2.
pip install PyYAML==5.3.1

3.
git clone https://github.com/tensorflow/models.git
copy <YOUR_PATH>\models\research\object_detection\packages\tf2 <YOUR_PATH>\models\research
python -m pip install --use-feature=2020-resolver .

4.
cd model\research
protoc object_detection/protos/*.proto --python_out=.

?.
pip install --upgrade protobuf==3.20.0

5.
cd ..
cd ..
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


more information how to run this application in:
C:\_WORK\GitHub\toolbox-scanning-api\flaskr\__init__.py


(27.12.2024 - Quick Fix)
SMALL ADJUSTMENTS IN SCRIPTS TO MAKE PROJECT RUN
(OTHERWISE IMPORT ERROR BECAUSE OF DEPRECATED TENSORFLOW VERSION OR SOMETHING)
class FreezableSyncBatchNorm(tf.keras.layers.BatchNormalization
# class FreezableSyncBatchNorm(tf.keras.layers.experimental.SyncBatchNormalization):

inside C:\_WORK\GitHub\toolbox-scanning-api\env\Lib\site-packages\object_detection\core\freezable_sync_batch_norm.py

from tensorflow.keras.preprocessing import image as image_ops
# from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops

inside C:\_WORK\GitHub\toolbox-scanning-api\env\Lib\site-packages\official\vision\image_classification\augment.py