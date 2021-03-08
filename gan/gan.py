from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import numpy as np

import argparse
import datetime

import os

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from common import load_data, save_image
from common import GANdalf as dalf

class GAN(dalf):
    def __init__(self, width, height,load_model=False):
        super().__init__()
        self.img_rows = height
        self.img_cols = width
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.model_save_dir = 'saved_model'
        self.last_checkpoint = 0
        self.model_loaded = load_model

        if load_model:
          self.load_model()
        else:
          self.init_model()

    def load_model(self):
      super().load_model()

      optimizer = Adam(0.0002, 0.5)

      # Load and compile the discriminator
      self.discriminator = self.load_keras_model("discriminator_model")
      self.discriminator.compile(loss='binary_crossentropy',
                                optimizer=optimizer,
                                metrics=['accuracy'])

      # Load the generator
      self.generator = self.load_keras_model("generator_model")

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = self.load_keras_model("combined_model")
      self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def init_model(self):
      optimizer = Adam(0.0002, 0.5)

      # Build and compile the discriminator
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                optimizer=optimizer,
                                metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

      self.write_stats_header()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))

        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def save_models(self):
        super().save_models()
        self.save_model(self.discriminator,
                    os.path.join(self.model_save_dir, 'discriminator_model'))
        self.save_model(self.generator,
                    os.path.join(self.model_save_dir, 'generator_model'))
        self.save_model(self.combined,
                    os.path.join(self.model_save_dir, 'combined_model'))

    def train(self, epochs, batch_size, sample_interval, save_interval, data_path):
        
        # Load the dataset
        X_train  = load_data(data_path)

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start=self.last_checkpoint if self.model_loaded else 0

        print('Training started')
        for epoch in range(start, epochs):
          
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            
            self.epochs.append({'accuracy':d_loss[1], 
                                'd_loss':d_loss[0],
                                'g_loss':g_loss})

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % save_interval == 0:
                self.last_checkpoint = epoch
                self.save_models()
                self.save_statistics()

            
        self.last_checkpoint =  epochs
        self.save_models()
        self.save_statistics()
      
    def generate_images(self, n, folder):
        super().generate_images(n,folder,'gan')
      

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--sample_interval',type=int,default=500)
    parser.add_argument('--save_interval',type=int,default=500)
    parser.add_argument('--test_data',type=str,default='')
    parser.add_argument('--load_model',default=False,action='store_true',dest='load_model')
    parser.add_argument('--generate_n',type=int,default=100)
    parser.add_argument('--epochs',type=int,default=10000)
    args = parser.parse_args()
    
    if args.mode == 'train':
        gan = GAN(256,256,args.load_model)
        gan.train(epochs=args.epochs, batch_size=32, sample_interval=500,save_interval=500,data_path='/content/drive/MyDrive/data/COVID-Net/positive/train')
    elif args.mode =='test':
        gan = GAN(256,256,load_model=True)
        gan.test('/content/drive/MyDrive/data/COVID-Net/positive/test')
    elif args.mode == 'generate':
        gan = GAN(256,256,load_model=True)
        gan.generate_images(n=args.generate_n, folder='/content/drive/MyDrive/data/COVID-Net/generated')
