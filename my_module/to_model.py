import numpy as np
import pandas as pd
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
import cloudpickle
from my_module.reconstruction_error import reconstruction_error_3d
from my_module.reconstruction_error import reconstruction_error_2d
from my_module.reconstruction_error import reconstruction_error_expert
from torch.autograd import Variable

def to_model_2d(input_img,index_num,input_size,model,error_th,AE_type):
    
    E_list = []
    judge_list = []
    train_model_list = []
    index_list = []
    list_data = []
    
    for i in range(index_num):
    
        save_input_img = input_img[i].detach().numpy()
        save_input_img = (save_input_img/2 +0.5)*255
        save_input_img = np.transpose(save_input_img,(1,2,0))

        if AE_type == "CONV":
            result_img = input_img[i]
        else:
            result_img = input_img[i].view(-1,input_size)
        
        result_img = Variable(result_img)
        #print(result_img.shape)
        result_img = model(result_img)
        
        
        result_img = (result_img/2 +0.5)*255
        result_np_img = result_img.detach().numpy()
       
        if AE_type != "CONV":
            result_np_img = result_np_img.reshape(1,28,28)
        
        result_np_img_reshape = np.transpose(result_np_img,(1,2,0)) 
        
        #print(result_np_img_reshape)
        
        r_error,judge = reconstruction_error_2d(save_input_img,result_np_img_reshape,error_th)
    
        E_list.append(r_error)
        judge_list.append(judge)
        train_model_list.append(error_th)
        index_list.append(i)
    
    list_data = [E_list,judge_list,train_model_list,index_list]
    df = pd.DataFrame(data = list_data)
    df_T = df.T
    df_T.to_csv("C:/Users/admin.H115/git/auto-encoder/csv/result.csv",mode ="a",index = False,
            header = False)

def to_model_3d(input_img,index_num,input_size,model,error_th,AE_type):

    E_list = []
    judge_list = []
    train_model_list = []
    index_list = []
    list_data = []

    for i in range(index_num):
        save_input_img = input_img[i].detach().numpy()
        save_input_img = (save_input_img/2 +0.5)*255
        save_input_img = np.transpose(save_input_img,(1,2,0))
        
        if AE_type == "CONV":
            result_img = input_img[i]
            result_img.detach().numpy()
            #result_img = cv2.GaussianBlur(result_img,(3,3),1.1) 
            result_img = np.expand_dims(result_img,0)
            result_img = torch.tensor(result_img)

        else:
            result_img = input_img[i].view(-1,input_size)
        
        result_img = Variable(result_img)
        result_img = model(result_img)
        
        
        result_img = (result_img/2 +0.5)*255 
        result_np_img = result_img.detach().numpy()
        
        if AE_type != "CONV":
            result_np_img = result_np_img.reshape(3,28,28)
        
        #print(result_np_img.shape)
        result_np_img_reshape = np.transpose(result_np_img,(1,2,0)) 
        
        #print(result_np_img_reshape)
        
        r_error,judge = reconstruction_error_3d(save_input_img,result_np_img_reshape,error_th)
        E_list.append(r_error)
        judge_list.append(judge)
        train_model_list.append(error_th)
        index_list.append(i)

    list_data = [E_list,judge_list,train_model_list,index_list]
    df = pd.DataFrame(data = list_data)
    df_T = df.T
    df_T.to_csv("C:/Users/admin.H115/git/auto-encoder/csv/result.csv",mode ="a",index = False,
            header = False)


def to_model_expert(input_img_color,input_img_lab,index_num,input_size,model,error_th,AE_type):


    E_list = []
    judge_list = []
    train_model_list = []
    index_list = []
    list_data = []
    with open("model/train_folder=color_train_retry image_type=color AE_type=COLOR optim=SGD lr=0.05 epoch=150.pkl","rb") as h:
        first_model = cloudpickle.load(h)
    
    for i in range(index_num):
        
        save_input_img_3c = input_img_color[i].detach().numpy()
        save_input_img_3c = (save_input_img_3c/2 +0.5)*255
        save_input_img_3c = np.transpose(save_input_img_3c,(1,2,0))
        
        if AE_type == "CONV":
            result_img = input_img_color[i]
            result_img.detach().numpy()
            #result_img = cv2.GaussianBlur(result_img,(3,3),1.1) 
            result_img = np.expand_dims(result_img,0)
            result_img = torch.tensor(result_img)

        else:
            result_img_3c = input_img_color[i].view(-1,3*28*28)
        
        result_img_3c = Variable(result_img_3c)
        result_img_out_3c = first_model(result_img_3c)
        
        result_img_out_3c = (result_img_out_3c/2 +0.5)*255
        result_np_img_out_3c = result_img_out_3c.detach().numpy()
        
        if AE_type != "CONV":
            result_np_img_out_3c = result_np_img_out_3c.reshape(3,28,28)
        
        #print(result_np_img.shape)
        result_np_img_reshape_3c = np.transpose(result_np_img_out_3c,(1,2,0)) 
        
        r_error,judge = reconstruction_error_expert(save_input_img_3c,result_np_img_reshape_3c)

#        if 8 <= r_error and r_error <= 12:
#            if error_th == "color":
#                result_img_3c = Variable(result_img_3c)
#                result_img_out_3c = model(result_img_3c)
#                result_img_out_3c = (result_img_out_3c/2 +0.5)*255
#                result_np_img_out_3c = result_img_out_3c.detach().numpy()
#                result_np_img_out_3c = result_np_img_out_3c.reshape(3,28,28)
#                result_np_img_reshape_3c = np.transpose(result_np_img_out_3c,(1,2,0)) 
#                r_error,judge = reconstruction_error_3d(save_input_img_3c,result_np_img_reshape_3c,error_th)
#            else:
#                save_input_img_1c = input_img_lab[i].detach().numpy()
#                save_input_img_1c = (save_input_img_1c/2 +0.5)*255
#                save_input_img_1c = np.transpose(save_input_img_1c,(1,2,0))
#                
#                result_img_1c = input_img_lab[i]
#                result_img_1c = result_img_1c.view(-1,28*28)
#                result_img_1c = Variable(result_img_1c)
#                result_img_1c = model(result_img_1c)
#                result_img_1c = (result_img_1c/2 +0.5)*255
#                result_np_img_1c = result_img_1c.detach().numpy()
#                
#                if AE_type != "CONV":
#                    result_np_img_1c = result_np_img_1c.reshape(1,28,28)
#                
#                result_np_img_reshape_1c = np.transpose(result_np_img_1c,(1,2,0)) 

#                r_error,judge = reconstruction_error_2d(save_input_img_1c,result_np_img_reshape_1c,error_th)

        E_list.append(r_error)
        judge_list.append(judge)
        train_model_list.append(error_th)
        index_list.append(i)

    list_data = [E_list,judge_list,train_model_list,index_list]
    df = pd.DataFrame(data = list_data)
    df_T = df.T
    df_T.to_csv("C:/Users/admin.H115/git/auto-encoder/csv/result.csv",mode ="a",index = False,
            header = False)
