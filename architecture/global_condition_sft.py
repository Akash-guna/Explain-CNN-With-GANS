import tensorflow as tf
from tensorflow.keras.layers import Conv2D,LeakyReLU,ReLU

class SFTLayer(tf.keras.Model):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = Conv2D(32,3,padding='same')#32
        self.SFT_scale_conv1 = Conv2D(64,3,padding='same')#64
        self.SFT_shift_conv0 = Conv2D(32,3,padding='same')#32
        self.SFT_shift_conv1 = Conv2D(64,3,padding='same')#64
        self.leaky_relu      = LeakyReLU(0.1)
    
    def call(self, x):
        scale = self.SFT_scale_conv1(self.leaky_relu(self.SFT_scale_conv0(x[1])))
        shift = self.SFT_shift_conv1(self.leaky_relu(self.SFT_shift_conv0(x[1])))
        return x[0] * (scale + 1) + shift

class ResBlock(tf.keras.Model):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.sft0  = SFTLayer()
        self.conv0 = Conv2D(64,3,padding='same')
        self.relu  = ReLU()
    def call(self,x,relu=True):
        features = self.sft0(x)
        features = self.conv0(features)
        if relu==True:
            features=self.relu(features)
        return (x[0] + features, x[1])

class ResBlockExtension(tf.keras.Model):
    def __init__(self):
        super(ResBlockExtension, self).__init__()
        self.resblocks= [ResBlock() for i in range(32)]
        self.SFTLayer= SFTLayer()
        self.conv1=Conv2D(64,3,padding='same')
    def call(self,X):
        type(X)
        for i in range(32):
            if i%2==0:
                X=self.resblocks[i](X)
            else:
                X=self.resblocks[i](X,relu=True)
        
        X= self.SFTLayer(X)
        X= self.conv1(X)
        return X

class SFT_NET(tf.keras.Model):
    def __init__(self):
        super(SFT_NET, self).__init__()
        self.conv0 = Conv2D(64,3,padding='same')
        self.sft_branch= ResBlockExtension()
        self.fin_conv = Conv2D(3,3,padding='same') 
        self.CondNet = tf.keras.Sequential(
        [Conv2D(128,4,strides=1,padding='same'), LeakyReLU(0.1) , Conv2D(128,1,padding='same'), 
        LeakyReLU(0.1),Conv2D(128,1,padding='same'), LeakyReLU(0.1),
        Conv2D(128,1,padding='same'), LeakyReLU(0.1),Conv2D(32,3,padding='same')]
        )
    
    def call(self,x):
        cond = self.CondNet(x[1])
        print(cond.shape)
        fea = self.conv0(x[0])
        res = self.sft_branch([fea, cond])
        fea = fea + res
        out = self.fin_conv(fea)
        return out

def create_global_sft(inp_shape1 =(256,256,3), inp_shape2=(256,256,8)):
    '''

    Inputs:
            inp_shape1  --> Shape of Input Image
            inp_shape2  --> Shape of Input Cumilative Gradcam Maps
    
    Outputs:
            model       --> The Global SFT Model.
    '''
    input1 = tf.keras.Input(inp_shape1)
    input2 = tf.keras.Input(inp_shape2)
    m= SFT_NET()([input1,input2])
    model= tf.keras.models.Model(inputs=[input1,input2],outputs=m)
    return model
    