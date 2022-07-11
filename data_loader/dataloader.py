import numpy as np

class DataGen():
    def __init__(self,batch_size,input_data,cgms,output_data):
        self.batch_size=batch_size
        self.input_data = input_data
        self.cgms = cgms
        self.output_data = output_data
        self.len_data = len(self.input_data)
        self.cur_start = 0
    
    def real_batch(self):
        out_batch = self.output_data[self.cur_start:self.cur_start+self.batch_size]
        i=0
        out_batch_2=[]
        while i<self.batch_size:
            if i+self.cur_start+1 < self.len_data:
                out_batch_2.append(self.output_data[self.cur_start+i+1])
            else:
                out_batch_2.append(self.output_data[i+1])
            i+=1
        
        inp_batch = self.input_data[self.cur_start:self.cur_start+self.batch_size]
        cgm_batch = self.cgms[self.cur_start:self.cur_start+self.batch_size]
        y_batch = [1 for i in range(self.batch_size)]
        return (np.array(inp_batch),np.array(cgm_batch),np.array(out_batch),np.array(out_batch_2),np.array(y_batch))
    
    def gen_batch(self,g_model):
        inp = self.input_data[self.cur_start:self.cur_start+self.batch_size]
        cgm_batch = self.cgms[self.cur_start:self.cur_start+self.batch_size]
        out_batch = self.output_data[self.cur_start:self.cur_start+self.batch_size]
        _,gen_data = g_model.predict_on_batch([inp,cgm_batch,out_batch])
        y_batch = [0 for i in range(self.batch_size)]
        return np.array(gen_data),np.array(y_batch)
    
    def update_batch(self):
        self.cur_start+=self.batch_size
        print(self.cur_start)
        if self.cur_start+self.batch_size>=self.len_data:
            self.cur_start=0        