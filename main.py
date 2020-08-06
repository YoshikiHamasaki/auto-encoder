from my_module import make_data_set as mds
from my_module import autoencoder as AE
from my_module import calculate as cal
from my_module import reconstruction_error as re
from my_module import show_bin_image as show_bin_image
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

num_epochs = 4
learning_rate = 0.001 #write train csv and test csv path 
out_dir = "result"    #write train image and test image path
input_size = 3*28*28


train_loader, test_loader = mds.bin_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/AE_train_bin","../image-data/AE_test_bad_bin")



def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


model = AE.bin_Autoencoder()


optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)


loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,f"result/epoch_{num_epochs}_model.pkl")

np.save('./{}/loss_list.npy'.format(out_dir), np.array(loss_list))
 
show_bin_image.show_image(test_loader,3,model)


#loss_list = np.load('{}/loss_list.npy'.format(out_dir))
#plt.plot(loss_list)
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.grid()
#plt.show()
