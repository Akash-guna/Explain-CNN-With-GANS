import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Concatenate,Add
class DiscriminatorBlock(tf.keras.Model):
    def __init__(self,params):
        super(DiscriminatorBlock,self).__init__()
        self.conv1= Conv2D(params,3,activation='relu',padding='same')
        self.conv2= Conv2D(params,3,activation='relu',padding='same')
        self.conv3= Conv2D(params,3,activation='relu',padding='same')
        self.maxpool = MaxPooling2D()
        self.add =Add()
    def call(self,X,res=False,maxpool=False):
            f = self.conv1(X)
            f = self.conv2(f)
            f = self.conv3(f)
            if res == True:
                if maxpool == True:
                    X_pool= self.maxpool(X)
                    f_pool= self.maxpool(f)
                    out=self.add([X_pool,f_pool])
                else:
                    out=self.add([X,f])
            else:
                out=f
            return out
    
class DiscriminatorBlockExtension(tf.keras.Model):
    def __init__(self,param):
        super(DiscriminatorBlockExtension,self).__init__()
        self.resblock1 = DiscriminatorBlock(param)
        self.resblock2 = DiscriminatorBlock(param)
        self.resblock3 = DiscriminatorBlock(param)
    def call(self,inp):
        X1=self.resblock1(inp)
        X1=self.resblock2(X1,res=True)
        X1=self.resblock3(X1,res=True,maxpool=True)
        return X1
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(32,activation='relu')
        self.d2 = Dense(16,activation='relu')
        self.d3 = Dense(1,activation='softmax')
        self.discblock1 = DiscriminatorBlockExtension(64)
        self.discblock2 = DiscriminatorBlockExtension(64)
        self.discblock3 = DiscriminatorBlockExtension(64)
        self.discblock4 = DiscriminatorBlockExtension(32)
        # self.discblock5 = DiscriminatorBlockExtension(16)
        self.conc = Concatenate()
        
    def call(self,inp):
        #resblock1 -inp 1
        X1 = self.discblock1(inp[0])
        #resblock2 -inp 2
        X2 = self.discblock2(inp[1])
        #resblock 3 - conc
        c = self.conc([X1,X2])
        X3 = self.discblock3(c)
        #resblock 4 - X3
        X4 = self.discblock4(X3)
        #resblock 5 - X4
        #X5 = self.discblock5(X4)
        flat = self.flatten(X4)#(X5)
        o = self.d1(flat)
        o = self.d2(o)
        o = self.d3(o)
        return o

def create_discriminator(inp1_shape=(256,256,3),inp2_shape=(256,256,3)):
    input1 = tf.keras.Input(inp1_shape)
    input2 = tf.keras.Input(inp2_shape)
    out = Discriminator()([input1,input2])
    model=tf.keras.models.Model(inputs=[input1,input2],outputs=out)
    return model