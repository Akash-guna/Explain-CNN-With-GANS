import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import applications as ap

models =[ap.InceptionV3,ap.ResNet50,ap.VGG16]
data_paths =['Datasets/Food']
model_paths= ['models/Food']
backbone_names =['InceptionV3',"ResNet","VGG16"]

for j,path in enumerate(data_paths):
    for i,backbone_model in enumerate(models):
        gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,validation_split=0.1)
        train=gen.flow_from_directory(path,target_size=(256,256),batch_size=64,subset='training')
        val=gen.flow_from_directory(path,target_size=(256,256),batch_size=64,subset='validation')
        
        m=backbone_model(
                    input_shape=(256,256,3),
                    include_top=False,
                    )
        
        inp=tf.keras.Input((64,64,3))
        m_out= m.output
        glob = tf.keras.layers.GlobalMaxPooling2D()(m_out)
        d1 = tf.keras.layers.Dense(256,activation='relu',kernel_initializer='he_normal')(glob)
        drop = tf.keras.layers.Dropout(0.1)(d1)
        d2= tf.keras.layers.Dense(128,activation='relu',kernel_initializer='he_normal')(d1)
        out= tf.keras.layers.Dense(len(np.unique(train.classes)),activation='softmax',kernel_initializer='he_normal')(d2)
        model = tf.keras.models.Model(inputs=m.input,outputs=out)
        opt = 'adam' if backbone_names[i]=='mobilenet' else 'sgd'
        print(f'Model = {backbone_names[i]}')
        print(f"Optimizer = {opt}")
        
        model.compile(optimizer=opt,loss='categorical_crossentropy',metrics='acc')
        try:
            os.makedirs(os.path.join(model_paths[j],backbone_names[i]))
        except:
            continue
        callbacks=[ tf.keras.callbacks.ModelCheckpoint(monitor='val_loss',filepath=os.path.join(model_paths[j],backbone_names[i],'Epoch={epoch:02d}- Loss={val_loss:.2f} - val_acc = {val_acc:.2f}.h5'))]
        history = model.fit(train,epochs=20,validation_data=val,callbacks=callbacks)