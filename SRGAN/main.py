import os
import io
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from IPython.display import FileLink
from keras.applications import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from networks import SRGAN
files = os.listdir("/kaggle/input/mirflickr25k/mirflickr25k/mirflickr")
files = [x for x in files if x.endswith(".jpg")]
files = files[:2500]

os.mkdir("LR")
os.mkdir("HR")

file_paths = [os.path.join("/kaggle/input/mirflickr25k/mirflickr25k/mirflickr", x) for x in files[:2500]]


# read and resize the images
def read_images(file_path, size, mode="lr"):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = (img - 127.5) / 127.5
    img = img.astype(np.float32)
    # if mode == "lr":
    # img = img /255.0
    # elif mode == "hr":

    # img = img / 255.0
    return img


for i, file_path in tqdm(enumerate(file_paths)):
    img_hr = read_images(file_path, 128, mode='hr')
    img_lr = read_images(file_path, 32, mode='lr')
    np.save("/kaggle/working/HR/img{}.npy".format(i), img_hr)
    np.save("/kaggle/working/LR/img{}.npy".format(i), img_lr)

# read the data
lr = os.listdir("/kaggle/working/LR")
hr = os.listdir("/kaggle/working/HR")

lr.sort()
hr.sort()

gen_paths = [os.path.join("/kaggle/working/LR", x) for x in lr]
disc_paths = [os.path.join("/kaggle/working/HR", x) for x in hr]


# read images
def read_images(file_paths):
    imgs = []
    for file_path in tqdm(file_paths):
        img = np.load(file_path)
        imgs.append(img)
    return np.array(imgs)


lr_images = read_images(gen_paths)
hr_images = read_images(disc_paths)


# lr_images=lr_images[:2500]
# hr_images = hr_images[:2500]
def denormalize_img(img):
    return ((img * 127.5) + 127.5).astype(np.uint8)


def mini_batches_(X, Y, batch_size=64):
    """
    function to produce minibatches for training
    :param X: input placeholder
    :param Y: mask placeholder
    :param batch_size: size of each batch
    :return:
    minibatches for training

    """
    gen_images = []
    disc_images = []
    train_length = len(X)
    num_batches = int(np.floor(train_length / batch_size))
    for i in tqdm(range(num_batches)):
        batch_x = X[i * batch_size: i * batch_size + batch_size]
        batch_y = Y[i * batch_size: i * batch_size + batch_size]
        batch_x = load_img(batch_x)
        batch_y = load_img(batch_y)
        gen_images.append(batch_x)
        disc_images.append(batch_y)
    return gen_images, disc_images

for i in range(10,20):
    plt.figure(figsize=(10, 10))
    plt.subplot(231)
    plt.title("\n\nLOW RESOLUTION(32)")
    plt.imshow(denormalize_img(lr_images[i]))
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(denormalize_img(hr_images[i]))
    plt.title("\n\nHIGH RESOLUTION(128)")
    plt.axis('off')

gan = SRGAN()
gan.compile(tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8),
            tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8))