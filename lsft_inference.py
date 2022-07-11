from data_loader.dataloader import DataGen
from data_loader.dataprocess import *
from architecture.arch_loader_lsft import discriminator,gan,generator
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

gen_model = generator()
disc = discriminator()
gan_model = gan(gen_model,disc)

gen_model.load_weights(r'models\LSFT_Strategy-2\gen\0\gan')
disc.load_weights(r'models\LSFT_Strategy-2\discriminator\0\disc')
gan_model.load_weights(r'models\LSFT_Strategy-2\gan\0\gan')

train,test=load_all_data()
inp_images_train,cgms_train,out_images_train,y=process_dataset(test)
gen=DataGen(16,inp_images_train,cgms_train,out_images_train)
inp_batch,cgm_batch,out_batch,out_batch_2,y_real = gen.real_batch()
_,gen_data = gan_model.predict_on_batch([inp_batch,cgm_batch,out_batch])

folder ='Inference-Food'
gen=DataGen(16,inp_images_train,cgms_train,out_images_train)
os.makedirs(f'Outputs/{folder}/gan')
os.makedirs(f'Outputs/{folder}/ori')
os.makedirs(f'Outputs/{folder}/inputs')
for j in range(25):
    inp_batch,cgm_batch,out_batch,out_batch_2,y_real = gen.real_batch()
    _,gen_data = gan_model.predict_on_batch([inp_batch,cgm_batch,out_batch])
    gen.update_batch()
    for i in range(gen_data.shape[0]):
        cv2.imwrite(f'Outputs/{folder}/gan/{j}_{i}.jpg',cv2.cvtColor(gen_data[i]*255,cv2.COLOR_RGB2BGR))
        plt.imsave(f'Outputs/{folder}/ori/{j}_{i}.jpg',out_batch[i])
        plt.imsave(f'Outputs/{folder}/inputs/{j}_{i}_{y_real[i]}.jpg',inp_batch[i])