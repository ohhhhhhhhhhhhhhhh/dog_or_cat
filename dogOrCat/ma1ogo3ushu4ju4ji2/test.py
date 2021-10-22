import keras
import os, random
import matplotlib
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from keras.preprocessing import image

def read_random_image():
    folder = r"ma1ogo3ushu4ju4ji2/dogs_cats/data/test/"
    file_path = folder + random.choice(os.listdir(folder))
    pil_im = Image.open(file_path, 'r')
    return pil_im

def get_predict(pil_im, model):
    # 图片缩放
    pil_im = pil_im.resize((200, 200))
    # 将图片转为 numpy array 格式
    array_im = image.img_to_array(pil_im)
    array_im = np.expand_dims(pil_im, axis=0)
    # 对图片进行预测
    result = model.predict([[array_im]])
    if result[0][0] > 0.5:
        print('预测结果：狗')
    else:
        print('预测结果：猫')

# model = keras.models.load_model('ma1ogo3ushu4ju4ji2/myModel.h5')
model = keras.models.load_model("ma1ogo3ushu4ju4ji2/dogs_cats/model/basic_cnn_model.h5")
pil_im = read_random_image()
pil_im.show()
#imshow(image.img_to_array(pil_im))
get_predict(pil_im, model)