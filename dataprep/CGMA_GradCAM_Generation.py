from .utils import get_img,all_img_paths
from .gradcam import *
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Parameters
MODEL_PATH = "../models/Food/VGG/Epoch=04- Loss=0.18 - val_acc = 0.96.h5"
DATASET_NAME = MODEL_PATH.split("/")[2]
ARCH_NAME = MODEL_PATH.split("/")[3]

#Load Image Paths
a,y =all_img_paths(f'../Datasets/{DATASET_NAME}')

#Split Into Test and Train
train_a, test_a,train_y, test_y = train_test_split(np.array(a), y, test_size=0.1, random_state=42,shuffle=False)
del y
del a
train_y = np.array(train_y)
test_y = np.array(test_y)

#One Hot Encode Y
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_y.reshape(-1,1))
train_y=enc.transform(train_y.reshape(-1,1)).toarray()
test_y=enc.transform(test_y.reshape(-1,1)).toarray()
cats=enc.categories_

#Load Classification Model 
model = tf.keras.models.load_model(MODEL_PATH)
model.layers[-1].activation = tf.keras.activations.linear

#Get All ConvLayers Information 
conv2D_layers = [layer.name for layer in reversed(model.layers) if len(layer.output_shape) == 4 and isinstance(layer, tf.keras.layers.Conv2D)]

# Make Directories Where the CGMS and GradCAMs would be Saved
os.makedirs(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}')
os.makedirs(f'../Gradcams/{DATASET_NAME}/{ARCH_NAME}')

# Number Of Train Images Available
o= len(train_a)

# BELOW is the CGMA Creation Loop

# Change o to any constant number ex. 100 to lesser the ammount of CGMAs generated . 
# Note:GradCAMs and CGMAs should be of equal quantity 
for i in range(o):
    os.makedirs(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}/'+str(i))
    img = get_img(train_a[i])
    p = model.predict(np.expand_dims(img,axis=0))
    p=np.argmax(p)
    #print(p)
    sp,cams=fuse_layers(conv2D_layers,model,cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR))
    c=None
    for j,cam in enumerate(cams):
        try:
            if c==None:
                pass
            c=cam
        except:
            c+=cam
        #print
        print(i ,end="\r")
        plt.imsave(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}/{i}/{j}.jpg',c,cmap='jet')

# BELOW is the GradCAM Creation Loop

# Change o to any constant number ex. 100 to lesser the ammount of CGMAs generated . 
# Note:GradCAMs and CGMAs should be of equal quantity 
for i in range(o):
    img = get_img(train_a[i])
    p = model.predict(np.expand_dims(img,axis=0))
    p=np.argmax(p)
    cam=gradcam_ll(conv2D_layers,model,cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR))
    plt.imsave(f'../Gradcams/{DATASET_NAME}/{ARCH_NAME}/{i}.jpg',cam,cmap='jet')
    print(i,end='\r')