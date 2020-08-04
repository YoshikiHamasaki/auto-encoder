#dataset作成関数　入力には訓練データとテストデータのcsvとそれぞれのパス入力
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import os
from torch import nn,optim
from torch.utils.data import Dataset
import pandas as pd


def make_data_set(train_csv,test_csv,train_img_path,test_img_path):

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
            image = Image.open( os.path.join(self.root_dir, image_name))
            #image = image.convert('RGB') # PyTorch 0.4以降
            # label (0 or 1)
            label = self.train_df.query('ImageName=="'+image_name+'"')['ImageLabel'].iloc[0]
            return self.transform(image), int(label)
    

    train_set = MyDataSet(train_csv, train_img_path)
    print(train_set[0])
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=32, shuffle=True)
    
    test_set = MyDataSet(test_csv,test_img_path)
    test_loader = torch.utils.data.DataLoader(
            test_set,batch_size= len(test_set) ,shuffle=False)
    
    return train_loader, test_loader
