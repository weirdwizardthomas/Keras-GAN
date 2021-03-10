import glob
import os

from PIL import Image
from matplotlib import cm

from tqdm import tqdm
import numpy as np

import datetime
import matplotlib.pyplot as plt

import random

from keras.models import model_from_json

class GANdalf:
  def __init__(self):
    self.model_save_dir = 'saved_model'
    self.model_loaded = False
    self.last_checkpoint = 0
    self.epochs = [] # todo need to save and load

    self.generator = ...
    self.stats_file = os.path.join(self.model_save_dir, 'training.csv')

  def write_stats_header(self):
    with open(self.stats_file,'w') as file: #clears file
      file.write('accuracy,g_loss,d_loss\n')
      
  def load_keras_model(self, model_name):
    json_name = os.path.join(self.model_save_dir, model_name + '.json')
    weights_name = os.path.join(self.model_save_dir, model_name + '.h5')

    with open(json_name, 'r') as json_file:
      loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_name)

    return loaded_model

  def save_models(self):
    print('Saving models')
    with open(os.path.join(self.model_save_dir,'checkpoint.txt'),'w') as file:
      file.write(str(self.last_checkpoint))

  def save_statistics(self):
    with open(self.stats_file,'a') as file:
      for epoch in self.epochs:
        line = '{accuracy},{g_loss},{d_loss}{delimiter}'.format(accuracy=epoch['accuracy'],g_loss=epoch['g_loss'],d_loss=epoch['d_loss'],delimiter='\n')
        file.write(line)

    self.epochs = []

  def save_model(self, model, model_path):
    print('Saving model {model_path}'.format(model_path=model_path))
    with open(str(model_path) + '.json', 'w') as json_file:
      json_file.write(model.to_json())

    model.save_weights(str(model_path + '.h5'))

  def load_model(self):
    filename = os.path.join(self.model_save_dir,'checkpoint.txt') 
    with open(filename,'r') as file:
      self.last_checkpoint = file.read()

  def generate_images(self, n, folder,model_name):
    noise = np.random.normal(0, 1, (n, self.latent_dim))
    generated_images = self.generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    for generated_image in tqdm(generated_images):
      img = (generated_image * 255).astype(np.uint8).reshape(256,256)
      filename = '{folder}/{model}_{timestamp}.png'.format(folder=folder,model=model_name,timestamp=datetime.datetime.now())
      save_image(img,filename)

  def sample_images(self, epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, self.latent_dim))
    gen_imgs = self.generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
          axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
          axs[i,j].axis('off')
          cnt += 1
      fig.savefig("images/%d.png" % epoch)
    plt.close()

  def test(self, test_folder):
    ...
  
  def train(self, epochs, batch_size, sample_interval,save_interval, data_path):
    ...

def load_data(folder, size=(256,256)):
  files = glob.glob('{folder}/*'.format(folder=folder))
 
  print('Loading images')
  counter = 0 
  images = []
  for file in tqdm(files):
    img = Image.open(file)
    if img.mode != 'L':
      img = img.convert(mode='L')
      counter += 1
    img = np.array(img.resize(size))
    images.append(img)
      
  print('Converted {counter} images'.format(counter=counter))
  return np.array(images)

def load_labeled_data(positive_folder, negative_folder, size=(256,256)):
  files = []
  labels = []

  positive_files = glob.glob('{folder}/*.*'.format(folder=positive_folder))
  negative_files = glob.glob('{folder}/*.*'.format(folder=negative_folder))

  dataset_size = min(len(positive_files), len(negative_files))

  positive_files = random.sample(positive_files,dataset_size)
  negative_files = random.sample(negative_files, dataset_size)
  
  files = negative_files + positive_files
  labels = [0] * dataset_size + [1] * dataset_size

  print('Loading images')

  counter = 0 
  images = []
  
  for file in tqdm(files):
    img = Image.open(file)
    if img.mode != 'L':
      img = img.convert(mode='L')
      counter += 1
    img = np.array(img.resize(size))
    images.append(img)
      
  print('Converted {counter} images'.format(counter=counter))

  return np.array(images), np.array(labels)

  

def save_image(image_array,filename):
  im = Image.fromarray(image_array)
  im.save(filename)