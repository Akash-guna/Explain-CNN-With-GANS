from data_loader.dataloader import DataGen
from data_loader.dataprocess import *
from architecture.arch_loader_gsft import discriminator,gan,generator
import tensorflow as tf

def train(gener,d_model, g_model,n_epochs=200, n_batch=1):
    #step 1 train the discriminator on Real Images
        #Step 1.1 Get Real Images Batch 
        print('LOADING DATA')
        train_data,test_data=load_all_data()
        print('PROCESSING DATA')
        inp_images_train,cgms_train,out_images_train,y_train= process_dataset(train_data)
        gen=DataGen(n_batch,inp_images_train,cgms_train,out_images_train)
        
        
        steps = inp_images_train.shape[0]/n_batch
        for epoch in range(n_epochs):
            print('Epoch = ',epoch)
            print("Steps = ",steps)
            for step in range(int(steps)):
                
                inp_batch,cgm_batch,out_batch,out_batch_2,y_real = gen.real_batch()
                fake_batch,y_fake=gen.gen_batch(g_model)
                gen.update_batch()
                if step %100 == 0:
                    d_loss1=d_model.train_on_batch([inp_batch,out_batch],y_real)       
                    d_loss2 = d_model.train_on_batch([inp_batch,fake_batch],y_fake)
                g_loss ,_,_ = g_model.train_on_batch([inp_batch,cgm_batch,out_batch],[y_real,out_batch])
                print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (step, d_loss1, d_loss2, g_loss))
            os.makedirs(f'models/LSFT/discriminator/{epoch}')
            os.makedirs(f'models/LSFT/gan/{epoch}')
            os.makedirs(f'models/LSFT/gen/{epoch}')
            d_model.save_weights(f'models/LSFT/discriminator/{epoch}/disc')
            gan_model.save_weights(f'models/LSFT/gan/{epoch}/gan')
            gener.save_weights(f'models/LSFT/gen/{epoch}/gan')

        return g_model  
gen_model = generator()
disc = discriminator()
gan_model = gan(gen_model,disc)

g_model=train(gen_model,disc,gan_model,n_epochs=25,n_batch=32)

