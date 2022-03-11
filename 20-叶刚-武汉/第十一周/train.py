"""
深度学习模型训练代码
"""
import tensorflow as tf
import numpy as np
import os
import utils
import matplotlib.pyplot as plt
import AlexNet
import VGG
import ResNet


# 设置日志级别，屏蔽通知信息和警告信息
# 0：默认值，输出所以信息；1：屏蔽通知信息；2：屏蔽通知信息和警告信息；3：屏蔽通知、警告和错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_train(dataset_folder, model_type='resnet', learning_rate=1e-4):
    """
    @param dataset_folder: 数据集根目录
    @param model_type: 网络模型的类型，取值 'alexnet'、'vgg'、'resnet'
    @param learning_rate: 学习率
    """
    # 数据集图像文件夹、类别id文件路径、标签文件路径
    img_folder = os.path.join(dataset_folder, 'image/train')
    classes_id_path = os.path.join(dataset_folder, 'model/index_word.txt')
    dataset_txt_path = dataset_folder + '/dataset.txt'

    # 解析类别id文件
    id_to_class, class_to_id = utils.parse_id_class_txt(classes_id_path)
    print('id_to_class: {}, class_to_id: {}'.format(id_to_class, class_to_id))

    classes = np.unique([key for key in class_to_id.keys()])
    num_classes = len(classes)
    print(f'classes: {classes}, num_classes: {num_classes}')

    # 读入标签文件，并将顺序打乱
    all_image_path, all_labels = utils.parse_and_shuffle_dataset(img_folder, dataset_txt_path)
    print('all_image_path[0]: {}, all_label_path[0]: {}'.format(all_image_path[0], all_labels[0]))

    # 数据集切片
    image_num = len(all_image_path)
    train_num = int(0.8 * image_num)
    test_num = image_num - train_num

    np.random.seed(2022)
    random_indices = np.random.permutation(image_num)   # 生成乱序indices
    all_image_path = np.array(all_image_path)[random_indices]
    all_labels = np.array(all_labels)[random_indices]

    train_images_path = all_image_path[:train_num]
    train_labels = all_labels[:train_num]
    test_images_path = all_image_path[train_num:]
    test_labels = all_labels[train_num:]

    # 创建训练集和测试集
    train_ds = tf.data.Dataset.from_tensor_slices((train_images_path, train_labels))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.map(utils.preprocess_train_dataset, num_parallel_calls=AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images_path, test_labels))
    test_ds = test_ds.map(utils.preprocess_test_dataset, num_parallel_calls=AUTOTUNE)

    # 产生源源不断的、乱序后的训练样本。每个batch样本数量为BatchSize
    train_ds = train_ds.repeat().shuffle(buffer_size=4*BatchSize).batch(BatchSize)
    # 产生测试样本，每个batch样本数量为BatchSize
    test_ds = test_ds.batch(BatchSize)

    # 创建模型
    if model_type == 'alexnet':
        model = AlexNet.AlexNet(input_shape=(224, 224, 3), num_classes=num_classes)
    elif model_type == 'vgg':
        model = VGG.VGG16(input_shape=(224, 224, 3), num_classes=num_classes)
    else:
        model = ResNet.ResNet50(input_shape=(224, 224, 3), num_classes=num_classes)
    model.summary()

    # 设置模型保存间隔
    save_folder = './checkpoint/' + model_type
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # save_path = save_folder + '/epoch-{epoch:03d}-val_acc-{val_acc:.4f}.h5'
    checkpoint_period = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_folder, 'epoch-{epoch:03d}-val_acc-{val_acc:.4f}.h5'),
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        peroid=2
    )

    # 设置学习率衰减策略，训练过程中连续5个epoch val_loss不降低，则将学习率减少为原来的一半
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    # 设置是否早停，训练过程中连续10个epoch val_loss不降低，则认为训练收敛，结束训练
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='auto',
    )

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['acc'])

    steps_per_epoch = train_num // BatchSize
    validation_step = test_num // BatchSize

    # 模型训练
    history = model.fit(train_ds,
                        batch_size=BatchSize,
                        epochs=Epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds,
                        validation_steps=validation_step,
                        callbacks=[checkpoint_period, reduce_lr, early_stop])

    # 模型评估
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'test_loss: {test_loss}, test_acc: {test_acc}')

    # 模型保存
    model.save(os.path.join(save_folder, model_type + '-cat_vs_dog-FinalModel.h5'))

    # 绘制曲线
    # history.history.keys()
    plt.plot(history.epoch, history.history.get('acc'), label='acc')
    plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
    plt.legend()

    plt.figure()
    plt.plot(history.epoch, history.history.get('loss'), label='loss')
    plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset_root_folder = '../00-data/datasets/cat_vs_dog'
    checkpoint_folder = './checkpoint'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    BatchSize = 16
    Epochs = 10
    LearningRate = 1e-4
    ModelType = 'resnet'    # 'alexnet', 'vgg', 'resnet'

    # 设置显存动态申请
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    run_train(dataset_root_folder, model_type=ModelType, learning_rate=LearningRate)

