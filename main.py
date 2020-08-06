from my_module import make_data_set as mds
from my_module import autoencoder as AE
from my_module import calculate as cal
from my_module import reconstruction_error as re
from my_module import pred_image as pred
import os
from PIL import Image
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
from torch import nn, optim
from torch.autograd import Variable

COLOR = 0b1
BINARY = 0b10

image_type = 0

#### setting parameter ######################
image_type |= COLOR
num_epochs = 50
learning_rate = 0.5
out_dir = "result"    
input_size = 3*28*28
train_csv_path = "csv/train.csv" 
test_csv_path = "csv/test_bad.csv" 
train_image_path = "../image-data/AE_train" 
test_image_path = "../image-data/AE_test_bad" 
device =  'cpu'
#############################################

def own_imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
        
if image_type & COLOR:
    save_type = "color"

    train_loader, test_loader = mds.color_make_data_set(
            train_csv_path,test_csv_path,train_image_path,test_image_path) 
    model = AE.color_Autoencoder(input_size) 
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) 
    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/{save_type}_lr_{lr}_epoch_{num_epochs}_model.pkl") 
    np.save(f"./{out_dir}/loss_list.npy", np.array(loss_list)) 
    iterator = iter(test_loader)
    input_img, _ = next(iterator)
    own_imshow(input_img[0])
    plt.show()
    
    result_img = input_img.view(input_img.size(0),-1)
    result_img = Variable(result_img)
    pred_img = model(result_img)
    
    pred = pred_img.reshape(-1,3,28,28)
    own_imshow(pred)

elif image_type & BINARY:
    save_type = "binary"
    
    train_loader, test_loader = mds.bin_make_data_set(
            train_csv_path,test_csv_path,train_image_path,test_image_path) 
    model = AE.bin_Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5) 
    loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,
            f"result/{save_type}_lr_{lr}_epoch_{num_epochs}_model.pkl") 
    np.save('./{}/loss_list.npy'.format(out_dir), np.array(loss_list)) 
    pred.pred_bin_image(test_loader,3,model) 
    #loss_list = np.load('{}/loss_list.npy'.format(out_dir))
    #plt.plot(loss_list)
    #plt.xlabel('iteration')
    #plt.ylabel('loss')
    #plt.grid()
    #plt.show()
else:
    print(select image_type)
