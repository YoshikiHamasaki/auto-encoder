from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

test_path = "C:/Users/admin.H115/git/image-data/test_shadow/171.jpg" 
test_path2 ="C:/Users/admin.H115/git/image-data/test_shadow/194.jpg"


train_loader1, test_loader1 = dataset.lab_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/train","../image-data/test_bad")


train_loader, test_loader = dataset.color_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/train","../image-data/test_bad")

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

iterator = iter(train_loader)
img, _ = next(iterator)
#imshow(img[2])

a = cv2.imread("C:/Users/admin.H115/git/image-data/8_12_train/166.jpg")
image = cv2.cvtColor(a,cv2.COLOR_BGR2HSV_FULL)

#print(hsv[:,:,2])
b = cv2.imread(test_path)
hsv1 = cv2.cvtColor(b,cv2.COLOR_BGR2HSV_FULL)
lab = cv2.cvtColor(b,cv2.COLOR_BGR2LAB)
b = cv2.cvtColor(b,cv2.COLOR_BGR2RGB)


plt.figure(figsize=(9,7))
plt.subplot(1,3,1)
plt.imshow(b)
plt.subplot(1,3,2)
plt.imshow(hsv1)
plt.subplot(1,3,3)
plt.imshow(lab)
plt.show()



#cv2.namedWindow('imagea', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('imagea', b)
#cv2.imshow('gammma', imga)
#cv2.waitKey()
