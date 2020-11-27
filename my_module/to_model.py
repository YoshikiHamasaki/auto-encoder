import numpy as np
import pandas as pd
import torch
import cv2
from my_module.reconstruction_error import reconstruction_error_3d
from my_module.reconstruction_error import reconstruction_error_2d
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
        first_model = "model/train_folder=color_train_retry image_type=color AE_type=COLOR optim=SGD lr=0.05 epoch=200.pkl"
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
        #print(result_img.shape)
        first_model = "model/train_folder=color_train_retry image_type=color AE_type=COLOR optim=SGD lr=0.05 epoch=200.pkl"
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


def to_model_expert(input_img,index_num,input_size,model,error_th,AE_type):


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
        #print(result_img.shape)
        first_model = "model/train_folder=color_train_retry image_type=color AE_type=COLOR optim=SGD lr=0.05 epoch=200.pkl"
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
