# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/9

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from assignment_AlexNet import AlexNet   #from后接的是文件名，文件的路径可以用.与文件隔开，import后直接接文件内的函数名,使用时可不带文件名；
import numpy as np
import assignment_utils    #只是导入文件，那么使用文件内的函数或方法时，就需要带上文件名，例如utils.resize_image(X_train,(224,224))
import cv2

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = assignment_utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)
        yield (X_train, Y_train)

if __name__ == '__main__':

    log_dir = "./logs/"  #设置模型存储路径

    with open(r"G:\八斗清华班\八斗清华班\【11】图像识别\代码\AlexNet-Keras-master\data\dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    model = AlexNet()   #创建模型

    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])


    batch_size = 128
    # steps_per_epoch=len(x_train)/batch_size
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])

    model.save_weights(log_dir + 'last1.h5')