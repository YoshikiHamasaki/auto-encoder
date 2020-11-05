import numpy as np
from my_module import reconstruction_error as recon
from torch.autograd import Variable

def to_model(input_img,index_num,input_size,model):
    
    for i in range(index_num):
    
        save_input_img = input_img[i].detach().numpy()
        save_input_img = (save_input_img/2 +0.5)*255
        save_input_img = np.transpose(save_input_img,(1,2,0))
        
        result_img = input_img[i].view(-1,input_size)
        result_img = Variable(result_img)
        #print(result_img.shape)
        result_img = model(result_img)
        
        
        result_img = (result_img/2 +0.5)*255
        result_np_img = result_img.detach().numpy()
        
        result_np_img = result_np_img.reshape(1,28,28)
        result_np_img_reshape = np.transpose(result_np_img,(1,2,0)) 
        
        #print(result_np_img_reshape)
        
        r_error = recon.bin_reconstruction_error(save_input_img,result_np_img_reshape)
        print(f"{r_error} index = {i}")
