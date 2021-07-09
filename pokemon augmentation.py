
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from numpy import expand_dims
from tensorflow.keras import layers
from tensorflow.keras import activations
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import os
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import math
import json
import random
import pprint
import scipy.misc
import imageio

from google.colab import drive
drive.mount('/content/drive')

def load_data():
  images = os.path.join("/content/drive/MyDrive/pokemon/pokemon/pokemon")
  ans = os.listdir(images)
  img = []
  path = '/content/drive/MyDrive/pokemon/pokemon/pokemon/'
  print(len(ans))
  j = 0
  for i in ans:
    temp = cv2.imread(path + i)
    temp1 = cv2.resize(temp, (128, 128))
    img.append(temp1)
    j += 1
    if j % 500 == 0:
      print(j)
  img = np.array(img)
  X = expand_dims(img, axis=-1)
  X = X.astype('float32')
  X = X / 255.0
  print(img.shape)
  return X

storing = load_data()

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(vertical_flip=True)
storing = np.reshape(storing, (storing.shape[0], storing.shape[1], storing.shape[2], storing.shape[3]))
print(storing.shape)
datagen.fit(storing)
# os.makedirs('images')
for X_batch in datagen.flow(storing, batch_size=storing.shape[0], save_to_dir='/content/drive/MyDrive/pokemon/pokemon/pokemon', save_prefix='aug', save_format='png'):
	# create a grid of 3x3 images
	for i in range(0, 9):
		pyplot.subplot(330 + 1 + i)
		# pyplot.imshow(X_batch[i].reshape(128, 128), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break

storing = load_data()

datagen = ImageDataGenerator(brightness_range=[0.5,1.0])
storing = np.reshape(storing, (storing.shape[0], storing.shape[1], storing.shape[2], storing.shape[3]))

it = datagen.flow(storing, batch_size=storing.shape[0], save_to_dir='/content/drive/MyDrive/pokemon/pokemon/pokemon', save_prefix='aug1', save_format='png')
for i in range(4):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
