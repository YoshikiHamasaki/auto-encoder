from my_module import makr_data_set as mds
from my_module import standard_AE as s_AE
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

num_epochs = 20       
learning_rate = 0.001 #write train csv and test csv path 
out_dir = "result"    #write train image and test image path

mds.make_data_set()

model = s_AE.Autoencoder()

def to_img(x):
    x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
    #x = x.clamp(0, 1)
    #x = x.view(x.size(0), 1, 28, 28)
    return x

device =  'cpu'


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=1e-5)


loss_list = []

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        x = img.view(img.size(0), -1)
        
        x = Variable(x)
        
        xhat = model(x)
    
        # $B=PNO2hA|!J:F9=@.2hA|!K$HF~NO2hA|$N4V$G(Bloss$B$r7W;;(B
        loss = criterion(xhat, x)
        #print("loss:{}".format(loss))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_list.append(loss.data.item())
    
    print('epoch [{}/{}], loss: {:.4f}'.format(
        epoch + 1,
        num_epochs,
        loss.data.item()))

np.save('./{}/loss_list.npy'.format(out_dir), np.array(loss_list))
torch.save(model.state_dict(), './{}/autoencoder.pth'.format(out_dir))
    
    
#loss_list = np.load('{}/loss_list.npy'.format(out_dir))
#plt.plot(loss_list)
#plt.xlabel('iteration')
#plt.ylabel('loss')
#plt.grid()
##plt.show()
test_img = test_img.view(test_img.size(0),-1)
test_img= Variable(test_img)
pred = model(test_img)
pred = pred.detach().numpy()
pred = to_img(pred)
print(pred)
pred = pred.reshape((28,28))



plt.subplot(2,1,1)
plt.imshow(test_img_print, cmap = "gray", vmin =0,vmax=1)

plt.subplot(2,1,2)
plt.imshow(pred, cmap = "gray", vmin = 0, vmax = 1)

plt.show()
