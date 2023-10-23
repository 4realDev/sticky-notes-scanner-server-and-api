# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro_sticky_notes_sync.py

#fmt: off
import nest_asyncio
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import os
import cv2
import numpy as np
import re

nest_asyncio.apply()

import config
#fmt: on

files = config.files
paths = config.paths


# VARS FOR REAL TIME OBJECT DETECTION
tf.config.run_functions_eagerly(True)
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(
    files['CUSTOM_MODEL_PIPELINE_CONFIG'])
detection_model = model_builder.build(
    model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(
    files['LABELMAP'])

min_score_thresh = config.min_score_thresh
bounding_box_and_label_line_thickness = config.bounding_box_and_label_line_thickness


# LOAD THE LATEST CHECKPOINT OF THE OBJECT DETECTION MODEL
# Necessary for any visual detection
def get_maximum_trained_model_checkpoint():
    maximum_ckpt_number = -1
    for file in os.listdir(paths['CUSTOM_MODEL_PATH']):
        try:
            file_ckpt_number = int(
                re.search(r'ckpt-(.*).index', file).group(1))
            if maximum_ckpt_number == -1:
                maximum_ckpt_number = file_ckpt_number
            if maximum_ckpt_number < file_ckpt_number:
                maximum_ckpt_number = file_ckpt_number
        except AttributeError:
            continue
    return maximum_ckpt_number


# Restore the latest checkpoint of trained model
# The checkpoint files correspond to snapshots of the model at given steps
# The latest checkpoint is the most trained state of the model
def load_latest_checkpoint_of_custom_object_detection_model():
    maximum_ckpt_number = get_maximum_trained_model_checkpoint()
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        paths['CUSTOM_MODEL_PATH'], f"ckpt-{maximum_ckpt_number}")).expect_partial()


# FUNCTIONS NECESSARY FOR REAL TIME OBJECT DETECTION
@ tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# use the detection model to preprocess, predict and postprocess image to get detections
# Uses the detection model inside TensorFlow detect_fn(image)
def get_detections_from_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.array(image)

    # when the lastest model checkpoint is not loaded,
    # the model will detect all objects wrong with low detection_score (0.0099)
    load_latest_checkpoint_of_custom_object_detection_model()

    # converting image into tensor object
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    # make detections on image
    detections = detect_fn(input_tensor)
    return detections


# returns a list with the relative ymin, xmin, ymax, xmax coordinates of those boundingboxes, which are above the min_score_thresh
def get_bounding_boxes_above_min_score_thresh(detections, imgHeight, imgWidth, min_score_thresh):
    formatted_scanned_object_detection_boxes = []
    for index, detection_score in enumerate(detections['detection_scores'][0]):
        if detection_score > min_score_thresh:
            scanned_object_detection_box = detections['detection_boxes'][0][index]
            scanned_object_class_index = int(
                detections['detection_classes'][0][index])
            scanned_object_class_color = config.labels[scanned_object_class_index]['color']
            scanned_object_class_name = config.labels[scanned_object_class_index]['name']

            # values of bounding-boxes must be treated as relative coordinates to the image instead as absolute
            ymin = round(float(scanned_object_detection_box[0] * imgHeight))
            xmin = round(float(scanned_object_detection_box[1] * imgWidth))
            ymax = round(float(scanned_object_detection_box[2] * imgHeight))
            xmax = round(float(scanned_object_detection_box[3] * imgWidth))

            # formated_scanned_object_label = list(
            #     filter(lambda label: (
            #         label['id'] == scanned_object_class), config.labels)
            # )

            formated_scanned_object_detection_box = {
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "classname": scanned_object_class_name,
                "color": scanned_object_class_color
            }

            # print(f"- bounding boxes (relative): {scanned_object_data} \n")

            formatted_scanned_object_detection_boxes.append(
                formated_scanned_object_detection_box)

    return formatted_scanned_object_detection_boxes


def get_bounding_box_color(detection_class: int):
    # yellow 122
    if detection_class == 1:
        return 122
    # blue 1
    elif detection_class == 2:
        return 1
    # pink 22
    elif detection_class == 3:
        return 22
    # green 48 / 94 / 120
    elif detection_class == 4:
        return 94
    else:
        return 0


# returns the given image with its overlay labeled bounding boxes from the passed detections (formatted with scores and label names)
# Uses viz_utils.visualize_boxes_and_labels_on_image_array from the object_detection/utils package from tensorflow to
# "Overlay labeled boxes on an image with formatted scores and label names."
def get_image_with_overlayed_labeled_bounding_boxes(
    image,
    detections,
    category_index=category_index,
    min_score_thresh=min_score_thresh,
    line_thickness=bounding_box_and_label_line_thickness,
):
    image_np = np.array(image)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    # necessary for detections['detection_classes']
    # because label ids start with 1 and detections['detection_classes'] start with 0
    label_id_offset = 1
    # make copy, because visualize_boxes_and_labels_on_image_array modifies image in place
    image_np_with_detections = image_np.copy()

    track_ids_list = []
    for detection_class in detections['detection_classes']+label_id_offset:
        track_ids_list.append(get_bounding_box_color(detection_class))

    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
        image=image_np_with_detections,
        boxes=detections['detection_boxes'],
        classes=detections['detection_classes']+label_id_offset,
        scores=detections['detection_scores'],
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        track_ids=np.array(track_ids_list),
        skip_track_ids=True,
        line_thickness=line_thickness)

    # if visualize_overlayed_labeled_bounding_boxes_in_image:
    #     # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    #     # plt.show()
    #     cv2.imshow("Visualize bounding-box detections in image",
    #                image_np_with_detections)
    #     # waits for user to press any key
    #     # (this is necessary to avoid Python kernel form crashing)
    #     cv2.waitKey(0)

    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()

    return image_np_with_detections
