"""
基于tensorflow.keras的AlexNet分类网络实现
tensorflow: 2.3.0
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


def AlexNet(input_shape: tuple = (224, 224, 3), num_classes: int = 1000):
    # 为了加快收敛，将每个卷积层的filter数量减半，全连接层减为1024、512
    model = tf.keras.Sequential([
        Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid',
               input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(128, (5, 5), (1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(192, (3, 3), (1, 1), padding='same', activation='relu'),
        Conv2D(192, (3, 3), (1, 1), padding='same', activation='relu'),
        Conv2D(128, (3, 3), (1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(num_classes)
    ])
    return model

# class AlexNet_1(tf.keras.Model):
#     def __init__(self, image_shape: tuple = (224, 224, 3), classes: int = 1000):
#         super(AlexNet_1, self).__init__()
#         self.image_shape = image_shape
#         self.classes = classes
#
#     def call(self, x):
#         model = tf.keras.Sequential([
#             Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
#                    input_shape=self.input_shape, activation='relu'),
#             MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
#             Conv2D(256, (5, 5), (1, 1), padding='same', activation='relu'),
#             MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
#             Conv2D(384, (3, 3), (1, 1), padding='same', activation='relu'),
#             Conv2D(384, (3, 3), (1, 1), padding='same', activation='relu'),
#             Conv2D(256, (3, 3), (1, 1), padding='same', activation='relu'),
#             MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
#             Flatten(),
#             Dense(4096, activation='relu'),
#             Dense(4096, activation='relu'),
#             Dense(self.classes)
#         ])(x)
#         return model
#
#
# class AlexNet(tf.keras.Model):
#     def __init__(self, input_shape: int = (224, 224, 3), classes: int = 1000):
#         super(AlexNet, self).__init__()
#         self.conv1 = Conv2D(input_shape=input_shape,
#                             filters=96,
#                             kernel_size=(11, 11),
#                             strides=(4, 4),
#                             padding='valid',
#                             activation='relu')
#
#         self.pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
#
#         self.conv2 = Conv2D(filters=256,
#                             kernel_size=(5, 5),
#                             strides=(1, 1),
#                             padding='same',
#                             activation='relu')
#
#         self.conv3 = Conv2D(filters=384,
#                             kernel_size=(3, 3),
#                             strides=(1, 1),
#                             padding='same',
#                             activation='relu')
#
#         self.flatten = Flatten()
#
#         self.fc1 = Dense(4096, activation='relu')
#
#         self.fc2 = Dense(classes, activation='softmax')
#
#     def call(self, x):
#         model = tf.keras.Sequential()
#         model.add(self.conv1,
#                   self.pool1,
#                   self.conv2,
#                   self.pool1,
#                   self.conv3,
#                   self.conv3,
#                   self.conv3,
#                   self.pool1,
#                   self.flatten,
#                   self.fc1,
#                   self.fc1,
#                   self.fc2)
#         output = model(x)
#         return output

