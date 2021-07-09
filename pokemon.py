
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

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')



def load_data():
  images = os.path.join("/content/drive/MyDrive/pokemon_new1")
  ans = os.listdir(images)
  img = []
  path = '/content/drive/MyDrive/pokemon_new1/'
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

def define_discriminator(in_shape=(128,128,3)):
  model = Sequential()
  model.add(Conv2D(32, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(256, (3,3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(512, (3,3), strides=(2, 2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(1, activation='sigmoid'))
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

# def define_discriminator(in_shape=(128,128,3)):
#   model = Sequential()
#   model.add(Conv2D(64, 4, strides=(2, 2), padding="same", use_bias=False, input_shape = in_shape))
#   model.add(BatchNormalization())
#   model.add(LeakyReLU(alpha = 0.2))

#   model.add(Conv2D(128, 4, strides=(2, 2), padding="same", use_bias=False))
#   model.add(BatchNormalization())
#   model.add(LeakyReLU(alpha = 0.2))

#   model.add(Conv2D(256, 4, strides=(2, 2), padding="same", use_bias=False))
#   model.add(BatchNormalization())
#   model.add(LeakyReLU(alpha = 0.2))

#   model.add(Conv2D(256, 4, strides=(2, 2), padding="same", use_bias=False))
#   model.add(BatchNormalization())
#   model.add(LeakyReLU(alpha = 0.2))

#   model.add(Conv2D(256, 4, strides=(2, 2), padding="same", use_bias=False))
#   model.add(BatchNormalization())
#   model.add(LeakyReLU(alpha = 0.2))

#   model.add(Conv2D(1, 4, strides=(1, 1), padding="valid", use_bias=False))
#   model.add(Flatten())
#   model.add(Dense(1, activation='sigmoid'))
#   opt = Adam(lr=0.00275, beta_1=0.5)
#   model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#   return model

def define_generator(latent_dim):
  model = Sequential()
  n_nodes = 4 * 4 * 512
  model.add(Dense(n_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((4, 4, 512)))
  model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(16, (4,4), strides=(2,2), padding='same'))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(3, (16,16), activation='sigmoid', padding='same'))
  return model

# def define_generator(latent_dim):
#   model = Sequential()
#   n_nodes = latent_dim * 1 * 1
#   model.add(Dense(n_nodes, input_dim = latent_dim))
#   model.add(Reshape((1, 1, 32)))
#   print('hii')
#   model.add(Conv2DTranspose(256, 4, strides = (1, 1), padding = "valid", use_bias = False))
#   print('hey')
#   model.add(BatchNormalization())
#   print('hi')
#   model.add(layers.Activation(activations.relu))
  
#   model.add(Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = "same", use_bias = False))
#   model.add(BatchNormalization())
#   model.add(layers.Activation(activations.relu))

#   model.add(Conv2DTranspose(256, (4, 4), strides = (2, 2), padding = "same", use_bias = False))
#   model.add(BatchNormalization())
#   model.add(layers.Activation(activations.relu))

#   model.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = "same", use_bias = False))
#   model.add(BatchNormalization())
#   model.add(layers.Activation(activations.relu))

#   model.add(Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = "same", use_bias = False))
#   model.add(BatchNormalization())
#   model.add(layers.Activation(activations.relu))

#   model.add(Conv2DTranspose(3, (4, 4), strides = (2, 2), padding = "same", use_bias = False))
#   model.add(layers.Activation(activations.tanh))
#   return model

a = define_discriminator()
a.summary()
b = define_generator(100)
b.summary()

def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.00275, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

def save_plot(examples, epoch, n=4):
  # print(examples[0, :, :, 0])
  examples = (examples * 255) // 1
  filename = '/content/drive/MyDrive/pokemon_output/_e%03d.png' % (epoch+1)
  # print(examples[0].shape)
  # data = Image.fromarray(examples[0], 'RGB')
  # data.save(filename)
  # save_images(examples, [128,128] ,filename)
  # cv2.imwrite(filename, img_tile)
  # pyplot.close()
  CONTROL_SIZE_SQRT = n
  WIDTH = 128
  HEIGHT = 128
  CHANNELS = 3
  control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
  # control_generated = generator.predict(control_vectors)
  for i in range(CONTROL_SIZE_SQRT ** 2):
      x_off = i % CONTROL_SIZE_SQRT
      y_off = i // CONTROL_SIZE_SQRT
      control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = examples[i, :, :, :]
  im = Image.fromarray(np.uint8(control_image * 255), 'RGB')
  im.save(filename)

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=16):
  # print('hi')
  X_real, y_real = generate_real_samples(dataset, n_samples)
  X_real = X_real.reshape((X_real.shape[0], X_real.shape[1], X_real.shape[2], X_real.shape[3]))
  _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
  x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
  # print(x_fake.shape)
  _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
  print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
  save_plot(x_fake, epoch)
  # filename = '/content/drive/MyDrive/pokemon_model_save/_%03d.h5' % (epoch + 1)
  # g_model.save(filename)

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=2000, n_batch=64):
  bat_per_epo = int(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)
  for i in range(n_epochs):
    for j in range(bat_per_epo):
      X_real, y_real = generate_real_samples(dataset, half_batch)
      X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      X_real = X_real.reshape((X_real.shape[0], X_real.shape[1], X_real.shape[2], X_real.shape[3]))
      # print(X_fake.shape, X_real.shape)
      X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
      d_loss, _ = d_model.train_on_batch(X, y)
      X_gan = generate_latent_points(latent_dim, n_batch)
      y_gan = ones((n_batch, 1))
      g_loss = gan_model.train_on_batch(X_gan, y_gan)
      print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
    if (i+1) % 10 == 0:
      summarize_performance(i, g_model, d_model, dataset, latent_dim)

latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_data()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

