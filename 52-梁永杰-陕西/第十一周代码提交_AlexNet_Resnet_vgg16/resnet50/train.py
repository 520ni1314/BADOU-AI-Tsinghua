from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np
import cv2

from model.resnet50 import Resnet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

# keras.json中设置的值，若从未设置过，则为“channels_last”。
K.image_data_format() == 'channels_first'

def array_from_file_generateor(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    # 我们的Keras生成器必须无限循环 ,每次需要一批新数据时，.fit_generator函数将调用我们的garray_from_file_generateor函数。
    while 1:
        # 在循环的每次迭代中，我们将我们的图像和标签重新初始化为空列表
        X_train = []
        Y_train = []

        # 得到一个batch大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)    # 清洗数据
            name = lines[i].split(';')[0]   # 读取图片文件名
            # 从文件中读取图像
            img = cv2.imread(r'./data/image/train'+'/'+name)  # r有清除\的转义字符功能
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # 调整图像大小
            img = cv2.resize(img, (224, 224))
            img = img/255

            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])    # 将标签存入
            # 读完一个周期开始
            i = (i+1) % n

        # 图像处理
        X_train = np.array(X_train)
        Y_train = to_categorical(np.array(Y_train),num_classes=2)  # num_classes 类别数
        # 但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行
        # 我们的生成器根据请求“生成”图像数组和调用函数标签列表
        yield (X_train, Y_train)    # 产生生成器变量，使用for，自动调用next()读取值


# if __name__ == "__main__":
#     数据生成函数的测试程序
#     with open("./data/dataset.txt", "r") as f:
#         lines = f.readlines()
#     for i in array_from_file_generateor(lines,2):
#         print(i[0].shape,i[1].shape)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = './logs/'
    # 打开数据集的txt
    with open('./data/dataset.txt','r') as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    np.random.seed(100)
    np.random.shuffle(lines)
    np.random.seed(None) # 清除种子

    # 90%用于训练，10%用于估计
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val


##############################################################################
    # 使用ResNet卷积作为特征提取，修改后2层分类层，重新训练做迁移学习
    Resnet50 = Resnet50()
    Resnet50.summary()

    weight_path = './logs/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    Resnet50.load_weights(weight_path)

    layer_fc1 = Dense(1024,activation='relu',name='fc1024')(Resnet50.layers[-2].output)
    layer_fc2 = Dense(2, activation='softmax', name='fc2')(layer_fc1)
    model = Model(Resnet50.input, layer_fc2,name='change_Resnet50')
    model.summary()

    # 设置所有层不做训练
    for layer in model.layers:
        layer.trainable = False

    model.layers[-1].trainalbe = True
    model.layers[-2].trainable = True

#################################################################################

    # 保存的保存训练模型参数，3代保存一次
    checkpoint_period1 = ModelCheckpoint(
            log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='acc',
            save_weights_only=False,
            save_best_only=True,
            period=3
    )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
            monitor='acc', # 监测的值，可以是accuracy，val_loss,val_accuracy
            factor=0.5,    # 缩放学习率的值，学习率将以lr = lr*factor的形式被减少
            patience=3,    # 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
            verbose=1      # 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
    )

    # 早停，当val_loss一直不下降的时候一位置模型基本训练完成，可以停止
    early_stopping = EarlyStopping(
            monitor='val_loss', # 监测的值，可以是accuracy，val_loss,val_accuracy
            min_delta=0,        # 增大或减小的阈值，
            patience=10,        # 能够容忍多少个epoch内都没有improvement。
            verbose=1           # 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
    )

    # 训练编译
    model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            metrics=['accuracy']   # 指标
    )

    # 一次的训练集大小
    batch_size = 128

    print("Train on {} sample,val on {} sample,with batch size {}.".format(num_train,num_val,batch_size))

    # 开始训练
    model.fit_generator(
            generator=array_from_file_generateor(lines[:num_train],batch_size),
            steps_per_epoch=max(1,num_train//batch_size), # “ // ” 表示整数除法，返回不大于结果的一个最大整数
            validation_data= array_from_file_generateor(lines[:num_val],batch_size),# 验证集
            validation_steps=max(1,num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period1,reduce_lr]
    )
    model.save_weights(log_dir+'last1.h5')