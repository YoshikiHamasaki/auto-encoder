import numpy as np
from my_module.reconstruction_error import reconstruction_error_3d
from my_module.reconstruction_error import reconstruction_error_2d
from torch.autograd import Variable

def to_model_2d(input_img,index_num,input_size,model,error_th):
    
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
        
        r_error = reconstruction_error_2d(save_input_img,result_np_img_reshape,error_th)
        print(f"{r_error} index = {i}")


def to_model_3d(input_img,index_num,input_size,model,error_th):
    
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
        
        result_np_img = result_np_img.reshape(-1,3,28,28)
        result_np_img_reshape = np.transpose(result_np_img,(1,2,0)) 
        
        #print(result_np_img_reshape)
        
        r_error = reconstruction_error_3d(save_input_img,result_np_img_reshape,error_th)
        print(f"{r_error} index = {i}")
