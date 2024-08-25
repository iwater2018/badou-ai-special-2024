import os
import keras.utils as np_utils
import numpy as np
import cv2
import keras.optimizers as Adam
from model.AlexNet import AlexNet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import utils

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while True:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像

            img = cv2.imread(os.path.join("data/image/train", name))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)
        yield (X_train, Y_train)


if __name__ =='__main__':
    log_dir = './logs/'
    with open(r"./data/dataset.txt","r") as f:
        lines = f.readlines()

    np.random.seed(10111)
    np.random.shuffle(lines)
    np.random.seed(None)
    #90% 用于训练， 10用于评估
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    #构建模型
    model = AlexNet()

    # 保存的方式，3世代保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='accuracy',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor = 'accuracy',
        factor = 0.5,
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

    model.complie(
        loss='categorical_crossentropy',  #损失函数
        optimizer=Adam(lr=1e-3),  # 使用 Adam 优化器，学习率为 1e-3
        metrice=['accuracy']   #评估指标为准确率
    )

    # 一次的训练集大小
    batch_size = 128

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'last1.h5')



