import matplotlib.image as mpimg 
import os 
  
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from tensorflow import keras 
from keras import layers 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling 
from sklearn.model_selection import train_test_split 
  
import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from glob import glob 
import cv2 
  
import warnings 
warnings.filterwarnings('ignore') 

# path to the folder containing our dataset 
dataset = './dataset/DATA'
  
# path of label file 
labelfile = pd.read_csv('./dataset/labels.csv') 

# load and preprocess dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.5, subset='training', image_size=(224, 224), seed=123, batch_size=32) 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.5, subset='validation', image_size=(224, 224), seed=123, batch_size=32) 

class_numbers = train_ds.class_names 
class_names = [] 
for i in class_numbers: 
    class_names.append(labelfile['Name'][int(i)]) 

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# data augmentation
data_augmentation = tf.keras.Sequential( 
    [ 
        #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(224, 224, 3)), 
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, input_shape=(224, 224, 3)), 
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, input_shape=(224, 224, 3)), 
        #tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical") 
    ] 
) 

# build neural network
model = Sequential() 
#model.add(data_augmentation) 
model.add(Rescaling(1./255, input_shape=(224, 224, 3))) 
model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
 
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 
model.add(Dense(256, activation='relu')) 
model.add(Dense(32, activation='relu'))  
model.add(Dense(len(labelfile), activation='softmax')) 
model.summary() 

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              optimizer='adam', 
              metrics=['accuracy']) 

# train neural network
mycallbacks = [EarlyStopping(monitor='val_loss', patience=5)] 
history = model.fit(train_ds, 
                 validation_data=val_ds, 
                 epochs=10, 
                 callbacks=mycallbacks) 


_, accuracy = model.evaluate(val_ds, batch_size=32, verbose=0)
print(accuracy)

model.save('model10.h5')

# Neural Network Model
#tf.keras.utils.plot_model(model, to_file='cnn.png', show_shapes=True)

# Loss 
plt.figure(0)
plt.plot(history.history['loss'],marker='o',label="Training Loss")
plt.plot(history.history['val_loss'],marker='s',label="Validation Loss")
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig('loss10.png')

# Accuracy 
plt.figure(1)
plt.plot(history.history['accuracy'],marker='o',label="Training Accuracy")
plt.plot(history.history['val_accuracy'],marker='s',label="Validation Accuracy")
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right') 
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(linestyle = '--', linewidth = 0.5)
plt.savefig('accuracy10.png')
