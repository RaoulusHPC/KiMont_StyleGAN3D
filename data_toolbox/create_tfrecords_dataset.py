import itertools
import threading
from pathlib import Path
from xml.dom import minidom
import queue
import scipy
import tensorflow as tf
import pandas as pd
import numpy as np
import os


import sys



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
from scipy.ndimage import zoom
import multiprocessing as mp
from tqdm import tqdm
from collections import deque
import traceback

######################

label_preposition = "XML"
labels_path = "../../labeled_parts/labels"
parts_path = "../../labeled_parts/parts"

######################



def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def serialize_example(part, label): #part
    encoded_part_bytes = tf.io.serialize_tensor(part)
    #part_bytes = tf.io.serialize_tensor(part)
    # attributes_bytes = tf.io.serialize_tensor(attributes)
    # name_bytes = tf.io.serialize_tensor(name)
    # category_bytes = tf.io.serialize_tensor(category)

    feature = {
        "component_raw": bytes_feature(encoded_part_bytes),
        "label_raw": float_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example






def write_to_dataset_worker(thread_number):
    record_file = './output_ranjit_dataset/comparator_IFC_dataset_part_{}.tfrecords'.format(thread_number)
    # parts_path = r"D:\data-toolbox\data\parts"
    global count
    with tf.io.TFRecordWriter(record_file, options) as writer:
        while not parts.empty():
            part = parts.get()

            try:
                data_filename = str(part).split("/")[-1]
                label_filename =label_preposition +  data_filename.replace('.npy', '.xml')
                label_filepath = os.path.join(labels_path, label_filename) #parts_path.replace("parts", "xml")


                if os.path.exists(label_filepath):
                    print("Label exists")
                    component1 = np.load(str(part))
                    component1 = np.squeeze(component1).astype('bool')
                    component1 = scipy.ndimage.binary_fill_holes(component1)
                    component1 = zoom(component1,(0.5,0.5,0.5))
                    
                    # component = img_as_bool(rescale(component, 0.5))
                    # label = np.load(label_filepath)[i].reshape((11))
                    xmldoc1 = minidom.parse(label_filepath)
                    itemlist1 = xmldoc1.getElementsByTagName('Bewertung')
                    label =float(itemlist1[3].firstChild.nodeValue)
                    print(label)

                    # if int(itemlist1[3].firstChild.nodeValue) == int(itemlist2[3].firstChild.nodeValue):
                    #     # label1 = 1.0
                    #     # label2 = 1.0
                    #     continue
                    # elif int(itemlist1[3].firstChild.nodeValue) > int(itemlist2[3].firstChild.nodeValue):
                    #     label1 = 1.0
                    #     label2 = 0.0
                    # else:
                    #     label1 = 0.0
                    #     label2 = 1.0


                    #
                    # value = int(itemlist[i].firstChild.nodeValue)
                    # current_label = np.zeros((1, 11))
                    # # current_label[0, value] = 1.0
                    # current_label = remaster(value)
                    # current_label = current_label.reshape((11))
                    # current_label = current_label.astype(np.float32)

                    
                    
                    serialized_example1 = serialize_example(component1  , label)
                    # serialized_example2 = serialize_example(component2, component1, label2)

                    

                    writer.write(serialized_example1)
                    # writer.write(serialized_example2)
                    
                    count += 1
                    print(str(count) + " / 255")
                    parts.task_done()
                else:
                    print("Failed")
                    print(label_filepath)
                    sys.exit()
            except Exception as e:
                print("Error: " + str(e))
                print(traceback.format_exc())
                break



if __name__ == '__main__':

    # from training.dataset import get_simplegrab_dataset
    # tfrecords_path = Path("output/")
    # binvox_filepaths = list(tfrecords_path.rglob('*.tfrecords'))
    # dataset = get_simplegrab_dataset(binvox_filepaths)
    # #dataset = iter(dataset)
    # count = 0
    # for i in dataset:
    #     print(str(count))
    #     count += 1

    tasks = queue.Queue()
    count = 0



    obj_filepaths = Path(str(parts_path)).rglob('*.npy')
    obj_filepaths = list(obj_filepaths) #[:10]
    #pairs = list((itertools.combinations(obj_filepaths, 2)))
    parts = queue.Queue()
    [parts.put(i) for i in obj_filepaths]


    options = tf.io.TFRecordOptions(compression_type='GZIP')


    for thread in range(16):
        threading.Thread(target=write_to_dataset_worker, args=(thread,)).start()
    print('waiting for tasks to complete')
    parts.join()
    print('done')