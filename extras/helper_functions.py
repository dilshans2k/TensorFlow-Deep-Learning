### Storing helper functions here which are constantly being used for deep learning notebooks


#Imports

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import zipfile
  
#Data Loading
def unzip_data(filename):
  zip_ref = zipfile.ZipFile(filename)
  zip_ref.extract_all()
  zip_ref


#Pre-Processing
def view_random_images(dirpath, classname):
  plt.figure(figsize = (10,7))
  complete_path = dirpath + classname
  random_images = random.sample(os.listdir(complete_path), 4)
  print(random_images)
  for i in range(4):
    plt.subplot(2,2,i+1)
    img = mpimg.imread(complete_path+'/'+random_images[i])
    plt.imshow(img)
    plt.title(classname)
    plt.axis("off")
    
    
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor and
  reshapes it to (img_shape, img_shape, color_channels)
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  #Resize the image
  img = tf.image.resize(img,size = [img_shape, img_shape])
  # Rescale the image (get all the values btw 0 and 1)
  img = img/255.
  return img


def better_prediction_viewing(model, input_image, true_label):
  img = load_and_prep_image(input_image)
  prediction = model.predict(tf.expand_dims(img, axis=0))
  if len(prediction[0]) > 1:
    pred_label = class_names[tf.argmax(prediction[0])]
  else:
    pred_label = class_names[int(tf.round(prediction))]
  plt.imshow(img)
  plt.title(f"Prediction: {pred_label}")
  plt.xlabel(f"Probability: {prediction}")
  


 
### transfer learning
#default model, just feature extraction
def create_model(model_url, num_classes = 10):
  """
  Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url(str): A TensorFlow Hub feature extraction URL.
    num_classes(int): Number of output neurons in the output layer,
      should be equal to the number of target classes, default 10.
  
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons
  """
  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable = False,
                                           name="feature_extraction_layer",
                                           input_shape = IMAGE_SHAPE + (3,)) #freeze the already learned layers
  
  # Create our own model
  model = tf.keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes, activation = 'softmax', name = 'output_layer')
  ])

  return model




### Callbacks
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


def plot_loss_curves(history):
  pd.DataFrame(history.history)['loss'].plot()
  pd.DataFrame(history.history)['val_loss'].plot()
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend()

  plt.figure()
  pd.DataFrame(history.history)['accuracy'].plot()
  pd.DataFrame(history.history)['val_accuracy'].plot()
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend()
  


def get_file_size(file_path):
  size = os.path.getsize(file_path)
  return size


def convert_bytes(size, unit = None):
  if unit=="KB":
    return print('File size: ' + str(round(size / 1024, 3)) + 'Kilobytes')
  elif unit == "MB":
    return print('File size: ' + str(round(size / (1024*1024), 3)) + 'Megabytes')
  else :
    return print('File size: ' + str(size) + 'bytes')
  
  
