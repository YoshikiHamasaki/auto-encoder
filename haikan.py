test
import os
from PIL import Image
import sys
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torchvision.transforms as transforms
import cv2
from torch import nn, optim
import glob

class MyDataSet(Dataset):
    def __init__(self, csv_path, root_dir):
        self.train_df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.images = os.listdir(self.root_dir)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 画像読み込み
        image_name = self.images[idx]
        image = Image.open( os.path.join(self.root_dir, image_name) )
        #image = image.convert('RGB') # PyTorch 0.4以降
        # label (0 or 1)
        label = self.train_df.query('ImageName=="'+image_name+'"')['ImageLabel'].iloc[0]
        return self.transform(image), int(label)

train_set = MyDataSet('train.csv', 'AE_train_bin')
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)



data_iter = iter(train_loader)
images, labels = data_iter.next()


img = np.array(images[0])
#print(img)
img = np.squeeze(img)
#print(img.shape)
img = img.reshape((28,28))
print(img)
plt.imshow(img,cmap="gray", vmin =0,vmax =1)
plt.show()                          
                            
class ConvAutoencoder(nn.Module):
    def __init__(self):   
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers   
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
                               kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 4,
                               kernel_size = 3, padding = 1)
        #Decoder Layers   
        self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
                                          kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
                                          kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
                          
    def forward(self, x): 
                          
        #encode#                          
        x = self.relu(self.conv1(x))        
        x = self.pool(x)                  
        x = self.relu(self.conv2(x))     
        x = self.pool(x)                  
        #decode#          
        x = self.relu(self.t_conv1(x))    
        x = self.sigmoid(self.t_conv2(x)) 
        return x          
                          

device =  'cpu'
net = ConvAutoencoder().to(device)

#def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam,
#              loss_fn = nn.MSELoss(), device = "cpu"):
#    """
#    n_epochs…訓練の実施回数
#    net …ネットワーク
#    device …　"cpu" or "cuda:0"
#    """
#    losses = []         #loss_functionの遷移を記録
#    optimizer = optimizer_cls(net.parameters(), lr = 0.001)
#    net.to(device)
#
#    for epoch in range(n_epochs):
#        running_loss = 0.0  
#        net.train()         #ネットワークをtrainingモード
#
#        for i, XX in enumerate(train_loader):
#            XX.to(device)
#            optimizer.zero_grad()
#            XX_pred = net(XX)             #ネットワークで予測
#            loss = loss_fn(XX, XX_pred)   #予測データと元のデータの予測
#            loss.backward()
#            optimizer.step()              #勾配の更新
#            running_loss += loss.item()
#
#        losses.append(running_loss / i)
#        print("epoch", epoch, ": ", running_loss / i)
#
#    return losses
#
#losses = train_net(n_epochs = 10,
#                   train_loader = train_loader,
#                   net = net)
