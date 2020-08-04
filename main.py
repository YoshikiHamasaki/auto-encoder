from my_module import make_data_set as mds
from my_module import standard_AE as s_AE
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

num_epochs = 10       
learning_rate = 0.001 #write train csv and test csv path 
out_dir = "result"    #write train image and test image path

train_loader, test_loader = mds.make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/AE_train_bin","../image-data/AE_test_bad_bin")


model = s_AE.Autoencoder()

def to_img(x):
    x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
    #x = x.clamp(0, 1)
    #x = x.view(x.size(0), 1, 28, 28)
    return x

device =  'cpu'


optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)


loss_list, model = cal.calculate(num_epochs,train_loader,model,optimizer,f"result/epoch_{num_epochs}_model.pkl")

np.save('./{}/loss_list.npy'.format(out_dir), np.array(loss_list))
 
    
#loss_list = np.load('{}/loss_list.npy'.format(out_dir))
#plt.plot(loss_list)
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.grid()
##plt.show()
data_iter = iter(test_loader)
images, labels = data_iter.next()
test_img = images[3]
in_img = test_img.reshape((28,28))
in_img = in_img.detach().numpy()
in_img = to_img(in_img)

out_img = test_img.view(test_img.size(0),-1)
out_img= Variable(out_img)
pred = model(out_img)
pred = pred.detach().numpy()
pred = to_img(pred)
pred = pred.reshape((28,28))

#print(pred)
#print(in_img)

r_error = re.reconstruction_error(in_img,pred)
print(r_error)

plt.subplot(2,1,1)
plt.imshow(in_img, cmap = "gray", vmin =0,vmax=1)

plt.subplot(2,1,2)
plt.imshow(pred, cmap = "gray", vmin = 0, vmax = 1)

plt.show()
