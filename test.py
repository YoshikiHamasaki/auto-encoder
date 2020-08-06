from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


train_loader, test_loader = dataset.color_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/AE_train","../image-data/AE_test_bad")

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

iterator = iter(train_loader)
img, _ = next(iterator)
imshow(img)
