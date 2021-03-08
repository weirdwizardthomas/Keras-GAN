from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

import argparse

import os

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

from common import load_data, save_image
from common import GANdalf as dalf

class WGAN(dalf):
    def __init__(self, width, height, load_model=False):
        super().__init__()
        self.img_rows = height
        self.img_cols = width
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.model_save_dir = 'saved_model'
        self.model_loaded = load_model
        self.critic = ...

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01

        if load_model:
            self.load_model()
        else:
            self.init_model()

    def load_model(self):
        optimizer = RMSprop(lr=0.00005)
        
        # Load and compile the critic
        self.critic = self.load_keras_model('critic_model')
        self.critic.compile(loss=self.wasserstein_loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # Load the generator
        self.generator = self.load_keras_model('generator_model')

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        validity = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, validity)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def init_model(self):
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        self.write_stats_header()

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        
        # 16x16
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        # 32x32
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # 64x64
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        # 128x128
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        # 256x256
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval, save_interval,data_path):

        # Load the dataset
        X_train = load_data(data_path)

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        start=self.last_checkpoint if self.model_loaded else 0

        for epoch in range(start,epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            self.epochs.append({'d_loss':1 - d_loss[0],
                                'g_loss': 1 -g_loss})
            
            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % save_interval == 0:
                self.last_checkpoint = epoch
                self.save_models()
                self.save_statistics()

        self.last_checkpoint = epochs
        self.save_models()
        self.save_statistics()

    def save_models(self):
        super().save_models()
        self.save_model(self.critic,
                        os.path.join(self.model_save_dir, 'critic_model'))
        self.save_model(self.generator,
                        os.path.join(self.model_save_dir, 'generator_model'))
        self.save_model(self.combined,
                        os.path.join(self.model_save_dir, 'combined_model'))
  
    def write_stats_header(self):
        with open(self.stats_file,'w') as file: #clears file
            file.write('g_loss,d_loss\n')
  
    def save_statistics(self):
        with open(self.stats_file,'a') as file:
            for epoch in self.epochs:
                line = '{g_loss},{d_loss}{delimiter}'.format(g_loss=epoch['g_loss'],d_loss=epoch['d_loss'],delimiter='\n')
                file.write(line)

        self.epochs = []

    def generate_images(self, n, folder):
      super().generate_images(n,folder,'wgan')
      

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--sample_interval',type=int,default=500)
    parser.add_argument('--save_interval',type=int,default=500)
    parser.add_argument('--test_data',type=str,default='')
    parser.add_argument('--load_model',default=False,action='store_true',dest='load_model')
    parser.add_argument('--generate_n',type=int,default=100)
    parser.add_argument('--epochs',type=int,default=4000)
    args = parser.parse_args()
   
    if args.mode == 'train':
        wgan = WGAN(256,256,load_model=args.load_model)
        wgan.train(epochs=args.epochs, batch_size=32, sample_interval=100, save_interval=500,  data_path='/content/drive/MyDrive/data/COVID-Net/positive/train')
    elif args.mode == 'test':
        wgan = WGAN(256,256,load_model=True)
        wgan.test('/content/drive/MyDrive/data/COVID-Net/positive/test')
    elif args.mode=='generate':
        wgan = WGAN(256,256,load_model=True)
        wgan.generate_images(n=args.generate_n, folder='/content/drive/MyDrive/data/COVID-Net/generated')