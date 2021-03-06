import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
 

def load_data():
  size = (256,256)
  files = glob.glob('/content/drive/MyDrive/data/COVID-Net/positive/*')
 
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
