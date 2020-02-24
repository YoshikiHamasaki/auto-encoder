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
from torch.autograd import Variable
import time
                      
num_epochs = 20       
learning_rate = 0.001 #write train csv and test csv path 
out_dir = "result"    #write train image and test image path

def data_load(train_csv,test_csv,train_img_path,test_img_path):

    class MyDataSet(Dataset):
        def __init__(self, csv_path, root_dir):
            self.train_df = pd.read_csv(csv_path)
            self.root_dir = root_dir
            self.images = os.listdir(self.root_dir)
            self.transform = transforms.Compose(
                    [transforms.ToTensor(),
                        transforms.Normalize((0.5,),(0.5,))])
            
        def __len__(self):
            return len(self.images)
    
        def __getitem__(self, idx):
            # 画像読み込みa
            image_name = self.images[idx]
            image = Image.open( os.path.join(self.root_dir, image_name) )
            #image = image.convert('RGB') # PyTorch 0.4以降
            # label (0 or 1)
            label = self.train_df.query('ImageName=="'+image_name+'"')
            ['ImageLabel'].iloc[0]
            return self.transform(image), int(label)
    
    train_set = MyDataSet('train_csv', 'train_img_path')
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=32, shuffle=True)
    
    test_set = MyDataSet("test_csv","test_img_path")
    test_loader = torch.utils.data.DataLoader(
            test_set,batch_size=2,shuffle=False)
    
    
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    
    test_img = images[1]
    
    test_img_print = test_img.reshape((28,28))

#img = np.squeeze(img)
##print(img.shape)
#img = img.reshape((28,28))
#print(img)
#plt.imshow(img,cmap="gray", vmin =0,vmax =1)
#plt.show()                          


class Autoencoder(nn.Module):
    

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()


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
    
        # 出力画像（再構成画像）と入力画像の間でlossを計算
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
#def train_net(n_epochs, train_loader, net, optimizer_cls = optim.Adam,
#              loss_fn = nn.MSELoss(), device = "cpu"):
#    """
#    n_epochs…訓練の実施回数
#    net …ネットワークa
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
