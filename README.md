# Pokemon-Generation
With every new generation of Pokémon, a whole slew of new species are introduced to the game, amounting to over 800 Pokémon species to-date! Wouldn't it be cool if we could use these Pokémon and train a model to generate new Pokémon for us?
## Overview
This repository is an implementation of DCGAN(Deep convolutional generative adversarial networks) architecture to generate Pokemons using Keras.
## Prerequisites
- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)
- Pokemon dataset from [Kaggle](https://www.kaggle.com/kvpratama/pokemon-images-dataset), or a static fallback provided by this. *Please note that if you use the original Kaggle version, it's better to switch the JPG images to PNG format (to avoid transparency handling later on)*

## Generative Adversarial Network (GAN)
GAN consist of two network:

 - A discriminator D receive input from training data and generated data. Its job is to learn how to distinguish between these two inputs.
 - A generator G generate samples from a random noise Z. Generator objective is to generate sample that is as real as possible it could not be distinguished by Discriminator.

### Deep Convolution GAN (DCGAN)
In DCGAN architecture, the discriminator `D` is Convolutional Neural Networks (CNN) that applies a lot of filters to extract various features from an image. The discriminator network will be trained to discriminate between the original and generated image. The process of convolution is shown in the illustration below:  
![](http://deeplearning.net/software/theano_versions/dev/_images/same_padding_no_strides_transposed.gif)

The network structure for the discriminator is given by:
<center>

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 128, 128, 3 | |
| convolution      | batch size, 64, 64, 32  | LRelu |
| convolution      | batch size, 32, 32, 64  |LRelu | 
| convolution      | batch size, 16, 16, 128  | LRelu |
| convolution      | batch size, 8, 8, 256 | LRelu |
| convolution      | batch size, 4, 4, 512 | LRelu |
| flatten     | batch size, 8192 | LRelu |
| dense      | batch size, 100 | Sigmoid |
| dense      | batch size, 1 | Sigmoid |

</center>

The generator `G`, which is trained to generate image to fool the discriminator, is trained to generate image from a random input. In DCGAN architecture, the generator is represented by convolution networks that upsample the input. The goal is to process the small input and make an output that is bigger than the input. It works by expanding the input to have zero in-between and then do the convolution process over this expanded area. The convolution over this area will result in larger input for the next layer. The process of upsampling is shown below:  
![](http://deeplearning.net/software/theano_versions/dev/_images/padding_strides_transposed.gif)

There are many name for this upsample process: full convolution, in-network upsampling, fractionally-strided convolution, deconvolution, or transposed convolution. 

The network structure for the generator is given by:

<center>

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 100 (Noise from uniform distribution) | |
| reshape layer      | batch size, 100, 1, 1  | Relu |
| deconvolution      | batch size, 512, 4, 4   |Relu | 
| deconvolution      | batch size, 256, 8, 8  | Relu |
| deconvolution      | batch size, 128, 16, 16 | Relu |
| deconvolution      | batch size, 64, 32, 32 | Relu |
| deconvolution      | batch size, 3, 64, 64 | Tanh |

</center>

### Hyperparameter of DCGAN
The hyperparameter for DCGAN architecture is given in the table below:

<center>

| Hyperparameter        |
| ------------- |
| Mini-batch size of 64     |
| Weight initialize from normal distribution with std = 0.02      |  
| LRelu slope = 0.2      |
| Adam Optimizer with learning rate = 0.0002 and momentum = 0.5      |

</center>
