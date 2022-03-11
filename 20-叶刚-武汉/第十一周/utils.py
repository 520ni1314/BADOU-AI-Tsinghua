"""
公共函数，如图像读取、resize等
"""
import numpy as np
import os
import tensorflow as tf


# 解析类别id文件
def parse_id_class_txt(txt_path):
    id_to_class = {}
    class_to_id = {}
    # 读入txt
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 解析txt内容
    for line in lines:
        ret = line.split('\n')[0].split(';')
        id_to_class[int(ret[0])] = ret[1]
        class_to_id[ret[1]] = int(ret[0])
    # print('id_to_class: {}, class_to_id: {}'.format(id_to_class, class_to_id))
    return id_to_class, class_to_id


# 解析数据集，打乱顺序
def parse_and_shuffle_dataset(img_folder, dataset_txt_path):
    all_image_path = []
    all_labels = []
    with open(dataset_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
    for line in lines:
        ret = line.split('\n')[0].split(';')
        all_image_path.append(os.path.join(img_folder, ret[0]))
        label_one_hot = tf.keras.utils.to_categorical(int(ret[1]), num_classes=2)
        all_labels.append(label_one_hot)
    return all_image_path, all_labels


# 读入训练图像，对图像进行预处理(归一化、图像增强)
def preprocess_train_dataset(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label


# 读入测试图像，对图像进行预处理
def preprocess_test_dataset(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image, label


# 预测时读入单张图像
def load_single_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)   # 单张图像要增加一个维度, batch维度 ---> (1, 224, 224, 3)
    return image

