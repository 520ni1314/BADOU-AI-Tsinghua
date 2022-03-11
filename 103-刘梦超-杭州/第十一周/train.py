#!/usr/bin/env python 
# coding:utf-8
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import np_utils
import utils
from AlexNetModel import AlexNet


# 数据生成器
def generator_data(lines, batch_size):
    i = 0
    while True:
        inputs = []
        labels = []
        for batch in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            # 图像名字
            name = lines[i].split(";")[0]
            # 通过名字加载图像
            img = cv2.imread("/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/data/image/train/" + name)
            img_nor = img / 255
            label = lines[i].split(";")[1]
            # 将图像保存到列表
            inputs.append(img_nor)
            # 将标签保存到列表
            labels.append(label)
            # 遍历整个数据后,重新开始
            i = (i + 1) % len(lines)
        # 调整图像尺寸
        inputs = utils.resize_img(inputs, (224, 224))
        # 将标签转换为独热编码
        labels = np_utils.to_categorical(labels, num_classes=2)
        yield inputs, labels


if __name__ == '__main__':
    # 模型保存位置
    log_dir = "/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/logs/"
    # 加载数据
    with open("/Users/lmc/Documents/workFile/AI/week11/test/AlexNet-Keras/data/dataset.txt", "r") as f:
        lines = f.readlines()
    # 总样本数
    total_num = len(lines)
    # 设置随机数种子,控制shuffle的随机性
    np.random.seed(10001)
    # 打乱数据
    np.random.shuffle(lines)
    np.random.seed(None)
    # 将数据分为训练集和测试集
    train_num = int(total_num * 0.9)
    validation_num = total_num - train_num
    # 加载模型
    model = AlexNet()
    # 设置优化器,损失函数,评价函数
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    # 设置早停
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )
    # 设置模型检查点
    model_checkpoint = ModelCheckpoint(
        log_dir + 'epoch{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        period=3
    )

    # 设置学习率下降
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
    )

    # 批大小
    batch_size = 128
    # 训练
    model.fit_generator(generator_data(lines[:train_num], batch_size),
                        steps_per_epoch=max(1, int(train_num / batch_size + 0.5)), epochs=50,
                        callbacks=[early_stop, model_checkpoint, reduce_lr],
                        validation_data=generator_data(lines[train_num:], batch_size),
                        validation_steps=max(1, int(validation_num / batch_size + 0.5)), initial_epoch=0)
    # 保存模型结果
    model.save_weights(log_dir + "last_result.h5")
