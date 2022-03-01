from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import cv2
"""
批量获取所有数据进行训练
"""
def get_train_data(lines,num_train):
    X_train = []
    Y_train = []
    #取前num_train行数据作为训练集
    for i in range(num_train):
        name = lines[i].split(';')[0]
        # 从文件中读取图像
        img = cv2.imread(r".\data\image\train" + '/' + name)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        X_train.append(img)
        Y_train.append(lines[i].split(';')[1])
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 224, 224, 3)
    Y_train = to_categorical(np.array(Y_train), num_classes=2)

    return X_train,Y_train



if __name__ == '__main__':

    # 模型保存的位置
    log_dir = "D:\\个人\\AI精品班\\10_八斗清华班\\【11】图像识别\\代码\\AlexNet-Keras-master\\logs\\"
    #打开数据集
    with open(r"D:\\个人\\AI精品班\\10_八斗清华班\\【11】图像识别\\代码\\AlexNet-Keras-master\\data\\dataset.txt","r") as f:
        lines = f.readlines()

    #打乱行
    np.random.seed(1000)
    np.random.shuffle(lines)
    np.random.seed(None)

    #80%用于训练
    num_val = int(len(lines)*0.9)
    num_train = len(lines)-num_val

    #建立AlexNet网络
    model = AlexNet()

    # 保存的方式，3代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'weights.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 交叉熵
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    callbacks_list = [checkpoint_period1,reduce_lr,early_stopping]

    train_images, train_labels = get_train_data(lines,num_train)
    # 训练模型
    #####################################################################
    model.fit(train_images, train_labels, epochs=10, batch_size=100,callbacks = callbacks_list)

    # model.save_weights(log_dir + 'last1.h5')


