"""
深度学习模型预测代码
"""

import tensorflow as tf
import utils
import AlexNet
import numpy as np
import os
import cv2 as cv


def predict_single_image(weights_path, img_path):
    # 1. 加载训练好的模型
    model = tf.keras.models.load_model(weights_path)
    # 2. 读入待测图像，对图像进行必要的处理
    img = utils.load_single_image(img_path)
    # 3. 将图像送入模型做推理，得到预测结果
    pred_output = model.predict(img)
    # 4. 对模型输出的结果进行解析处理，得到类别
    pred_class = id_to_class[np.argmax(pred_output)]
    return pred_class


if __name__ == '__main__':
    ModelType = 'vgg'  # 'alexnet', 'vgg', 'resnet'

    weights_path = './checkpoint/' + ModelType + '/' + ModelType + '-cat_vs_dog-FinalModel.h5'

    dataset_root_folder = '../00-data/datasets/cat_vs_dog'
    image_folder = os.path.join(dataset_root_folder, 'image/train/')
    classes_id_path = os.path.join(dataset_root_folder, 'model/index_word.txt')

    # 解析类别id文件
    id_to_class, class_to_id = utils.parse_id_class_txt(classes_id_path)
    print('id_to_class: {}, class_to_id: {}'.format(id_to_class, class_to_id))

    # 从样本集中随机选取一个样本
    img_files = os.listdir(image_folder)
    index = int(np.random.randint(0, len(img_files), 1))
    img_path = os.path.join(image_folder, img_files[index])

    # 对选取的样本进行预测
    predict_class = predict_single_image(weights_path, img_path)
    actual_class = img_path.split('/')[-1].split('.')[0]
    print(f'img_path: {img_path}, actual_class: {actual_class}, predict_class: {predict_class}')

    # 样本可视化
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    cv.imshow('img', img)
    cv.waitKey(0)

