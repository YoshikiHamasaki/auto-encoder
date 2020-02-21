import os
from PIL import Image
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available!')

num_epochs = 20
batch_size = 128
learning_rate = 0.001
out_dir = './autoencoder'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [0,1] => [-1,1]
])
MNIST_dataset = MNIST('./data', download=True, transform=img_transform)

#train_data_1 = train_dataset()
train_loader = DataLoader(MNIST_dataset, batch_size=batch_size, shuffle=False)

# matplotlibで1つ目のデータを可視化してみる
data_iter = iter(train_loader)
images, labels = data_iter.next()

data_0 = images[labels == 0]
data_0 = data_0.reshape([data_0.shape[0],1,28,28])
data_0_train, data_0_test =  data_0[:6000,:,:,:], data_0[:6000,:,:,:]

print(len(train_loader))
# npimg = data_0_train[15].numpy()
# npimg = npimg.reshape((28, 28))
# plt.imshow(npimg, cmap='gray')
# plt.show()

args = sys.argv

img = np.array( Image.open(args[1]) )
print(img.shape)
print(img[0])



# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#         #Encoder Layers
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16,
#                                kernel_size = 3, padding = 1)
#         self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 4,
#                                kernel_size = 3, padding = 1)
#         #Decoder Layers
#         self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
#                                           kernel_size = 2, stride = 2)
#         self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
#                                           kernel_size = 2, stride = 2)
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         #コメントに28×28のモノクロ画像をi枚を入力した時の次元を示す
#         #encode#                          #in  [i, 1, 28, 28] 
#         x = self.relu(self.conv1(x))      #out [i, 16, 28, 28]  
#         x = self.pool(x)                  #out [i, 16, 14, 14]
#         x = self.relu(self.conv2(x))      #out [i, 4, 14, 14]
#         x = self.pool(x)                  #out [i ,4, 7, 7]
#         #decode#
#         x = self.relu(self.t_conv1(x))    #out [i, 16, 14, 14]
#         x = self.sigmoid(self.t_conv2(x)) #out [i, 1, 28, 28]
#         return x

# net = ConvAutoencoder().to("cpu")

# def train_net(n_epochs, train_loader, net, optimizer_cls = torch.optim.Adam,
#               loss_fn = nn.MSELoss(), device = "cpu"):
#     """
#     n_epochs…訓練の実施回数
#     net …ネットワーク
#     device …　"cpu" or "cuda:0"
#     """
#     losses = []         #loss_functionの遷移を記録
#     optimizer = optimizer_cls(net.parameters(), lr = 0.001)
#     net.to(device)

#     for epoch in range(n_epochs):
#         running_loss = 0.0  
#         net.train()         #ネットワークをtrainingモード

#         for i, XX in enumerate(train_loader):
#             XX.to(device)
#             optimizer.zero_grad()
#             XX_pred = net(XX)             #ネットワークで予測
#             loss = loss_fn(XX, XX_pred)   #予測データと元のデータの予測
#             loss.backward()
#             optimizer.step()              #勾配の更新
#             running_loss += loss.item()

#         losses.append(running_loss / i)
#         print("epoch", epoch, ": ", running_loss / i)

#     return losses

# losses = train_net(n_epochs = 30,
#                    train_loader = train_loader,
#                    net = net)
