r""" 
    0.	Activate virtual environment
    cd [ROOT_PATH]
    .\venv\Scripts\activate

    1.	Adjust the “config.py” file in according to your needs
        If need model is uploaded, adjust the CUSTOM_MODEL_NAME_SUFFIX, PRETRAINED_MODEL_NAME. num_steps
        
        If new label is added, adjust the labels

    2.	Start GUI with 
        python gui.py
        
    3.  Test Model with data provided in test_data_set with:
        python test_model_on_data_set.py
"""

import os
# labels the custom model detects
labels = [
    {'name': 'yellow_sticky_note', 'id': 1, 'color': 'yellow'},
    {'name': 'blue_sticky_note', 'id': 2, 'color': 'blue'},
    {'name': 'pink_sticky_note', 'id': 3, 'color': 'red'},
    {'name': 'green_sticky_note', 'id': 4, 'color': 'green'}
]

# vars for real time object detection
min_score_thresh = 0.99
bounding_box_and_label_line_thickness = 2
save_original_image_overlayed_with_labeled_detection_bounding_boxes = True
save_original_image_overlayed_with_text_bounding_boxes = True
save_image_overlayed_with_ocr_visualization = True

# necessary for cmd commands
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CUSTOM_MODEL_NAME_SUFFIX = '2_own_min_data'
PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
num_steps = 25000
CUSTOM_MODEL_NAME = f'{CUSTOM_MODEL_NAME_SUFFIX}_{PRETRAINED_MODEL_NAME}_{str(num_steps)}'

# file names
LABEL_MAP_NAME = 'label_map.pbtxt'
PIPELINE_CONFIG_NAME = 'pipeline.config'


# paths to folders
paths = {
    'MIRO_TIMEFRAME_SNAPSHOTS': os.path.join(BASE_PATH, 'miro-timeframe-snapshots'),
    'ANNOTATION_PATH': os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'annotations'),
    'CUSTOM_MODEL_PATH': os.path.join(BASE_PATH, 'workspace', CUSTOM_MODEL_NAME, 'models', CUSTOM_MODEL_NAME),
}

# paths to files
files = {
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME),
    'CUSTOM_MODEL_PIPELINE_CONFIG': os.path.join(paths['CUSTOM_MODEL_PATH'], PIPELINE_CONFIG_NAME),
}
