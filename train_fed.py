import matplotlib.image as mpimg 
import os 
  
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from keras.utils import to_categorical
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential, load_model 
from keras import layers 
from tensorflow import keras 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling 
from sklearn.model_selection import train_test_split 
  
import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from glob import glob 
import cv2 

import seaborn as sns
from sklearn.metrics import classification_report
  
import warnings 
warnings.filterwarnings('ignore') 

# path to the folder containing our dataset 
dataset = './dataset/DATA'
  
# path of label file 
labelfile = pd.read_csv('./dataset/labels.csv') 

def load_models(num_models):
    models = []
    for i in range(0, num_models):
        models.append(load_model(f'model{i+1}.h5'))
    return models

def aggregate_weights(weights, models):
    average_model_weights = []
    n_models = len(models)
    n_layers = len(models[0].get_weights())
    for layer in range(n_layers):
        layer_weights = np.array([model.get_weights()[layer] for model in models])
        average_layer_weights = np.average(layer_weights, axis=0, weights=weights)
        average_model_weights.append(average_layer_weights)
    return average_model_weights

# plot confusion matrix
def plot(true, pred, labels, target_names):
    clf_report = classification_report(true, pred, labels=labels, target_names=target_names, output_dict=True)

    plt.figure()
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig('clf_report.png')

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

# Get and set weights for global model
weights = [93.72, 92.04, 87.77, 92.95, 89.44, 92.47, 93.38, 90.94, 91.65, 90.60] # change for each time ran with accuracies from generated models
x = max(weights)
idx = weights.index(x)
weights[idx] = 1
for i in range(len(weights)):
    if(weights[i] != 1):
        #weights[i] = 0.01       
        weights[i] = 0.02/(len(weights)-1)

models = load_models(len(weights))
average_weights = aggregate_weights(weights, models)
model.set_weights(average_weights)    

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              optimizer='adam', 
              metrics=['accuracy']) 

_, accuracy = model.evaluate(val_ds, batch_size=32, verbose=0)
print(accuracy)

# Plot Neural Network Model
#tf.keras.utils.plot_model(model, to_file='cnn.png', show_shapes=True)

# Save Model
model.save('global_model10.h5')


