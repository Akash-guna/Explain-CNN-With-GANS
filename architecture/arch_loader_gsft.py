
from .global_condition_sft import create_global_sft
from .discriminator import create_discriminator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def generator():
    gen = create_global_sft(inp_shape1 =(64,64,3), inp_shape2=(64,64,8))
    return gen
def discriminator():
    disc = create_discriminator(inp1_shape=(64,64,3),inp2_shape=(64,64,3))
    disc.compile(loss='binary_crossentropy',optimizer='adam',loss_weights=0.3)
    return disc
def gan(gen,disc):
    disc.trainable= False
    input_img = tf.keras.Input((64,64,3))
    input_cgm = tf.keras.Input((64,64,8))
    disc_input = tf.keras.Input((64,64,3))
    gen_out = gen([input_img,input_cgm])
    
    disc_out = disc([disc_input,gen_out])
    gan_model = tf.keras.models.Model(inputs=[input_img,input_cgm,disc_input],outputs=[disc_out,gen_out])
    opt = Adam(lr=0.0002, beta_1=0.5)
    gan_model.compile(loss=['binary_crossentropy','mae'], optimizer=opt,loss_weights=[1,100])
    return gan_model
    