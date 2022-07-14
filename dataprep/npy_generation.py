
from .utils import pos_list,all_img_paths
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

DATASET_NAME = "Food"
ARCH_NAME = "VGG"

l=len(os.listdir(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}/0'))-1
o = len(os.listdir(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}'))

n=8
pos=pos_list(8,l)
print(pos)
cgms=[]
for i in range(o):
    cgm_row=[]
    #print(i)
    for p in pos:
            try:
                img=cv2.imread(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}/{i}/{p}.jpg')
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(64,64))
                cgm_row.append(img)
            except:
                print(f'../CGMS/{DATASET_NAME}/{ARCH_NAME}/{i}/{p}.jpg')
    cgms.append(cgm_row)

grads=[]
for i in range(o):
    img=cv2.imread(f'../Gradcams/{DATASET_NAME}/{ARCH_NAME}/{i}.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(64,64))
    grads.append(img)

a,y =all_img_paths(f'{DATASET_NAME}/{ARCH_NAME}')
train_a, test_a,train_y, test_y = train_test_split(np.array(a), y, test_size=0.1, random_state=42,shuffle=False)

# 1 row has (input_image,array of CGMs,final gradcam output,y)
data=[]
for count,i in enumerate(range(o)):
    inp = cv2.imread(train_a[i])
    inp=cv2.cvtColor(inp,cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp,(64,64))
    tup=np.array([inp,np.array(cgms[count]),grads[count],train_y[i]])
    data.append(tup)
    print(count,end='\r')

data=np.array(data)
np.random.seed(7)
np.random.shuffle(data)

np.save(f'../npys/{DATASET_NAME}_{ARCH_NAME}.npy',data)





