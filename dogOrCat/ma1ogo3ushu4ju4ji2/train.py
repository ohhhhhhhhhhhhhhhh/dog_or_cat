from keras import layers, models, utils, optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

def define_cnn_model():

    model = models.Sequential()
    # 卷积层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
    # 最大池化层
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten层
    model.add(layers.Flatten())
    # 全连接层
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # 编译模型
    opt = optimizers.SGD(lr=0.03, momentum=0.9)
    model.compile(
        optimizer = opt,
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return model

def train_cnn_model():
    model = define_cnn_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        "ma1ogo3ushu4ju4ji2/dogs_cats/data/train",
        class_mode = 'binary',
        batch_size = 64,
        target_size = (200, 200)
    )
    model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=30, verbose=1)
    return model

print('文件当前路径：', os.getcwd())
print('开始训练')
model = train_cnn_model()
print('训练结束')
print('开始存储模型')
model.save('ma1ogo3ushu4ju4ji2/myModel.h5')
print('模型存储完毕')