from my_module import make_data_set as mds
from my_module import autoencoder as AE
from my_module import calculate as cal
from my_module import pred_image as pred
from my_module import select_optimizer_type as select_optim
from my_module import select_AE_type as select_AE
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Variable



def for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,input_size,train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name,error_th):

    device =  "cpu" 
    save_type = "color"
    
    train_loader, test_loader = mds.color_make_data_set(
            train_csv_path,test_csv_path,train_image_path,test_image_path) 

    model = select_AE.select_AE_type(AE_type,input_size)

    optimizer = select_optim.select_optimizer_type(optimizer_type,model,learning_rate)

    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/train_folder={train_name} image_type={save_type} AE_type={AE_type} optim={optimizer_type} lr={learning_rate} epoch={num_epochs}.pkl") 

    np.save(f"./result/loss_list.npy", np.array(loss_list)) 
    
    pred.pred_color_image(test_loader,test_img_index,model,input_size,error_th)

def for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name,error_th):

    device =  'cpu'
    save_type = "binary"
    
    train_loader, test_loader = mds.bin_make_data_set(
       train_csv_path,test_csv_path,train_image_path,test_image_path) 

    model = select_AE.select_AE_type(AE_type,input_size)
    
    optimizer = select_optim.select_optimizer_type(optimizer_type,model,learning_rate,weight_decay)

    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/train_folder={train_name} image_type={save_type} optim={optimizer_type} lr={learning_rate} epoch={num_epochs}.pkl") 
    np.save(f"./result/loss_list.npy", np.array(loss_list)) 
    pred.pred_bin_image(test_loader,test_img_index,model,error_th) 
    #loss_list = np.load('{}/loss_list.npy'.format(out_dir))
    #plt.plot(loss_list)
    #plt.xlabel('iteration')
    #plt.ylabel('loss')
    #plt.grid()
    #plt.show()

def for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name,error_th):

    device =  'cpu'
    save_type = "lab"
    
    train_loader, test_loader = mds.lab_make_data_set(
       train_csv_path,test_csv_path,train_image_path,test_image_path) 

    model = select_AE.select_AE_type(AE_type,input_size)
    
    optimizer = select_optim.select_optimizer_type(optimizer_type,model,learning_rate)

    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/train_folder={train_name} image_type={save_type} optim={optimizer_type} lr={learning_rate} epoch={num_epochs}.pkl") 

    np.save(f"./result/loss_list.npy", np.array(loss_list)) 
    pred.pred_lab_image(test_loader,test_img_index,model,input_size,error_th) 

