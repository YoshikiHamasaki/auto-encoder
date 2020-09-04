import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os
from torch import nn,optim
from torch.utils.data import Dataset
import pandas as pd
import cv2

#dataset作成関数　入力には訓練データとテストデータのcsvとそれぞれのパス入力
def bin_make_data_set(train_csv,test_csv,train_img_path,test_img_path):

    class my_data_set(Dataset):
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
            image = Image.open( os.path.join(self.root_dir, image_name))
            #image = image.convert('RGB') # PyTorch 0.4以降
            # label (0 or 1)
            label = self.train_df.query('ImageName=="'+image_name+'"')['ImageLabel'].iloc[0]
            return self.transform(image), int(label)
    

    train_set = my_data_set(train_csv, train_img_path)


    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=10, shuffle=True)
    
    test_set = my_data_set(test_csv,test_img_path)
    test_loader = torch.utils.data.DataLoader(
            test_set,batch_size= len(test_set) ,shuffle=False)
    
    return train_loader, test_loader


def color_make_data_set(train_csv,test_csv,train_img_path,test_img_path):

    class my_data_set(Dataset):
        def __init__(self, csv_path, root_dir):
            self.train_df = pd.read_csv(csv_path)
            self.root_dir = root_dir
            self.images = os.listdir(self.root_dir)
            self.transform = transforms.Compose(
                    [transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            
        def __len__(self):
            return len(self.images)
    
        def __getitem__(self, idx):
            # 画像読み込みa
            image_name = self.images[idx]
            image = Image.open( os.path.join(self.root_dir, image_name))
            image = image.convert('RGB') # PyTorch 0.4以降
            # label (0 or 1)
            label = self.train_df.query('ImageName=="'+image_name+'"')['ImageLabel'].iloc[0]
            return self.transform(image), int(label)
    

    train_set = my_data_set(train_csv, train_img_path)


    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=10, shuffle=True)
    
    test_set = my_data_set(test_csv,test_img_path)
    test_loader = torch.utils.data.DataLoader(
            test_set,batch_size= len(test_set) ,shuffle=False)
    
    return train_loader, test_loader

def lab_make_data_set(train_csv,test_csv,train_img_path,test_img_path):

    class my_data_set(Dataset):
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
            image = cv2.imread( os.path.join(self.root_dir, image_name))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            image_lab = image[:,:,0]
            print(image_lab.ndim)
            #image = image.convert('RGB') # PyTorch 0.4以降
            # label (0 or 1)
            label = self.train_df.query('ImageName=="'+image_name+'"')['ImageLabel'].iloc[0]
            return self.transform(image_lab), int(label)
    

    train_set = my_data_set(train_csv, train_img_path)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=10, shuffle=True)
    
    test_set = my_data_set(test_csv,test_img_path)
    test_loader = torch.utils.data.DataLoader(
            test_set,batch_size= len(test_set) ,shuffle=False)
    
    return train_loader, test_loader
