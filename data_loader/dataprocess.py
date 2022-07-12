import numpy as np
import os
import glob
import cv2
import tensorflow as tf

def load_all_data():
    np.random.seed(7)
    files = glob.glob('npys/*.npy')
    print(files)
    data=np.concatenate([np.load(i,allow_pickle=True)[:5000]for i in files])
    np.random.shuffle(data)
    l= data.shape[0]
    split=int(np.floor(0.1*l))
    test  = data[:split]
    train = data[split:]
    return (train,test)

def process_dataset(data):
    inp_images    = np.array(data[:,0].tolist())/255.
    #cgms          = np.array(data[:,1].tolist())/255.
    #cgms          = np.array([cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)/255. for j in i for i in data[:1].tolist()])
    cgms=[]
    count=0
    for i in np.array(data[:,1].tolist()):
        li=[]
        for j in i:
            li.append(cv2.cvtColor(j,cv2.COLOR_RGB2GRAY)/255.)
        print(count,end='\r')
        count+=1
        cgms.append(li)
    cgms=np.array(cgms)    
    cgms=tf.constant(cgms)
    cgms=tf.transpose(cgms,perm=[0,2,3,1]).numpy()
    out_images    = np.array(data[:,2].tolist())/255.
    y             = np.array(data[:,3].tolist())
    print('cgms shape = ',cgms.shape)
    return (inp_images,cgms,out_images,y)