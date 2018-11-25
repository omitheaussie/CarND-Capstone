import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib as pl
from PIL import Image
import string_int_label_map_pb2
from google.protobuf import text_format
import glob

# What model to download.
MODEL_NAME = 'trafficsignal'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'trafficsignal.pbtxt')

NUM_CLASSES = 1

IMAGE_SIZE = (12, 8)

TEST_IMAGE_PATHS = (glob.glob("images/*.jpg"))
#print(TEST_IMAGE_PATHS)
print(TEST_IMAGE_PATHS)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print(detection_graph)

def load_labelmap(path):
  with tf.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def _validate_label_map(label_map):
  for item in label_map.item:
    if item.id < 1:
      raise ValueError('Label map ids should be >= 1.')
    
def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
  categories = []
  list_of_ids_already_added = []
  if not label_map:
    label_id_offset = 1
    for class_id in range(max_num_classes):
      categories.append({
          'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)
      })
    return categories
  for item in label_map.item:
    if not 0 < item.id <= max_num_classes:
      logging.info('Ignore item %d since it falls outside of requested '
                   'label range.', item.id)
      continue
    if use_display_name and item.HasField('display_name'):
      name = item.display_name
    else:
      name = item.name
    if item.id not in list_of_ids_already_added:
      list_of_ids_already_added.append(item.id)
      categories.append({'id': item.id, 'name': name})
  return categories

def create_category_index(categories):
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

label_map = load_labelmap(PATH_TO_LABELS)
categories = convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = create_category_index(categories)