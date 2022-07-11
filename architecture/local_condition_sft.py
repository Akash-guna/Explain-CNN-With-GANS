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
        self.conv1 = Conv2D(64,3,padding='same')
        self.conv2 = Conv2D(128,3, padding='same')
        self.conv3 = Conv2D(64,3, padding='same')
        self.relu  = LeakyReLU()
        
    def call(self,x,SFT=False,relu=False):
        
        if SFT ==True:
            features = self.sft0(x)
            features = self.conv0(features)
            if relu==True:
                features=self.relu(features)
            return (x[0] + features)
        
        else:
            features = self.conv1(x[0])
            features = self.conv2(features)
            features = self.conv3(features)
            features = self.relu(features)
            return (x[0] + features)

class ResBlockExtension(tf.keras.Model):
    def __init__(self):
        super(ResBlockExtension, self).__init__()
        self.resblocks= [ResBlock() for i in range(16)]
        self.conv1=Conv2D(64,3,padding='same')

    def call(self,X,conds):
        j=0
        for i in range(15):
            c = [conds[k][j][0] for k in range(len(conds))]
            c=tf.convert_to_tensor(c)
            
            if i%2==0:
                X=self.resblocks[i]([X,c],SFT=True)    
                j+=1
            else:
                X=self.resblocks[i]([X,c],relu=True)
        X= self.conv1(X)
        return X

class SFT_NET(tf.keras.Model):
    def __init__(self,batch_size,n_cond):
        super(SFT_NET, self).__init__()
        self.conv0 = Conv2D(64,3,padding='same')
        self.sft_branch= ResBlockExtension()
        self.CondNet = tf.keras.Sequential(
        [Conv2D(128,4,strides=1,padding='same'), LeakyReLU(0.1) , Conv2D(128,1,padding='same'), 
        LeakyReLU(0.1),Conv2D(128,1,padding='same'), LeakyReLU(0.1),
        Conv2D(128,1,padding='same'), LeakyReLU(0.1),Conv2D(32,3,padding='same')])
        
        self.batch_size = batch_size
        self.n_cond = n_cond
        self.fin_conv=Conv2D(3,3,padding='same')
    
    def call(self,x):
        #print(x[1].shape)
        d=tf.transpose(x[1],perm=[0,3,1,2])
        #print(d.shape)
        conds=[[self.CondNet(tf.expand_dims(tf.expand_dims(d[i][j],axis=0),axis=-1)) for j in range(self.n_cond)] for i in range(self.batch_size)]
        #print(len(conds),len(conds[0]),conds[0][0].shape)
        fea = self.conv0(x[0])
        res = self.sft_branch(fea, conds)
        fea = fea + res
        out =self.fin_conv(fea)
#         cond = self.CondNet(x[1])
#         fea = self.conv0(x[0])
#         res = self.sft_branch([fea, cond])
#         fea = fea + res
#         out = Conv2D(3,3,padding='same')(fea)
        return out

def create_local_sft(input_shape1=(256,256,3),input_shape2=(256,256,8),batch_size=5,n_conds=8):
    '''

    Inputs:
            inp_shape1  --> Shape of Input Image
            inp_shape2  --> Shape of Input Cumilative Gradcam Maps
            batch_size  --> Batch Size
            n_conds     --> Number of Conditional Maps Provided
    
    Outputs:
            model       --> The Global SFT Model.
    '''
    input1 = tf.keras.Input(input_shape1)
    input2 = tf.keras.Input(input_shape2)
    m= SFT_NET(batch_size,n_conds)([input1,input2])
    model= tf.keras.models.Model(inputs=[input1,input2],outputs=m)
    return model