from my_module import make_data_set as mds
from my_module import autoencoder as AE
from my_module import calculate as cal
from my_module import reconstruction_error as re
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




#### setting parameter #####
decide_dim = "color"
num_epochs = 10       
learning_rate = 0.5
out_dir = "result"    
input_size = 3*28*28
############################




train_loader, test_loader = mds.color_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/AE_train","../image-data/AE_test_bad")



def own_imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()



model = AE.color_Autoencoder()


device =  'cpu'


optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


loss_list, model = cal.calculate(num_epochs,train_loader,model(input_size),optimizer,
        f"result/{decide_dim}_epoch_{num_epochs}_model.pkl")

np.save('./{}/loss_list.npy'.format(out_dir), np.array(loss_list))
 

iterator = iter(test_loader)
input_img, _ = next(iterator)
own_imshow(input_img[0])
plt.show()

result_img = input_img.view(input_img.size(0),-1)
result_img = Variable(result_img)
pred_img = model(result_img)

pred = img.reshape(-1,3,28,28)
own_imshow(pred)
