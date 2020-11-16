from torch.autograd import Variable
import torch
import torchvision
from PIL import Image
from torch import nn,optim
import cloudpickle
import os

def calculate(num_epochs,train_loader,model,optimizer,model_pass,AE_type):

 criterion = nn.MSELoss()
 loss_list = []
 mycheck = os.path.isfile(model_pass)

 if not mycheck: 
     for epoch in range(num_epochs):
         for data in train_loader:
             img, _ = data

             if AE_type == "CONV":
                 x = img
             else:
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

     with open(model_pass,"wb") as f:
        cloudpickle.dump(model,f)

 with open(model_pass,"rb") as f:
     model = cloudpickle.load(f)

 return loss_list, model
