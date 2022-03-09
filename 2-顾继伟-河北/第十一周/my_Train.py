from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import Adam
from my_AlexNet import my_AlexNet

import numpy as np
import my_Utils
import cv2
from keras import backend as K
K.image_data_format() == "channel_first"


def generate_arrays_from_file(lines,batch_size):
    n = len(lines)
    i = 0
    while 1:
        x_train = []
        y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
                name = lines[i].split(';')[0]
                img = cv2.imread(r".\data\image\train" + '/' + name)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = img/255
                x_train.append(img)     #x_train添加图片
                y_train.append(lines[i].split(';')[1])      #y_train添加标签
                i = (i+1)%n     #一轮训练完毕
        x_train = my_Utils.resize_image(x_train,(224,224))
        x_train = x_train.reshape(-1,224,224,3)
        y_train = np_utils.to_categorical(np.array(y_train),num_classes=2)
        yield(x_train,y_train)

if __name__ == '__main__':
    logdir = "./logs/"

    with open(r".\data\dataset.txt","r") as f:      #标记文件
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_Val = int(len(lines)*0.1)
    num_Train = int(len(lines)*0.9)

    model = my_AlexNet()        #调用模型

    check_point = ModelCheckpoint(
        logdir + 'history.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,     #acc3次不下降就不再降低学习率
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,        #val_loss10次不下降即停止训练
        verbose=1
    )

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy']
                  )
    batch_size = 128
    print("train on {} samples,val on {} samples,with batch_size{}".format(num_Train,num_Val,batch_size))

    model.fit_generator(generate_arrays_from_file(lines[:num_Train],batch_size),
                        steps_per_epoch=max(1,num_Train//batch_size),
                        validation_data=generate_arrays_from_file(lines[num_Train:],batch_size),
                        validation_steps=max(1,num_Val/batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=([check_point,reduce_lr])
    )
    model.save_weights(logdir+'last1.h5')



