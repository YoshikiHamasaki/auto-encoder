from torch.autograd import Variable
import torch
import torchvision
from PIL import Image
from torch import nn,optim


def calculate(num_epochs,train_loader,model,optimizer):

 criterion = nn.MSELoss()
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

 return loss_list, model
