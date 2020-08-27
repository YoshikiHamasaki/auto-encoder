from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


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
hsv = cv2.cvtColor(a,cv2.COLOR_BGR2HSV_FULL)

#plt.imshow(hsv)
#plt.show()
#print(hsv[:,:,0])
b = cv2.imread("C:/Users/admin.H115/git/image-data/test_shadow/112.jpg")
hsv2 = cv2.cvtColor(b,cv2.COLOR_BGR2HSV_FULL)

#plt.imshow(hsv2)
#plt.show()
#print(hsv2[:,:,0])

#c = cv2.imread("C:/Users/admin.H115/git/image-data/8_12_train/166.jpg")
#lab = cv2.cvtColor(c,cv2.COLOR_BGR2LAB)
#print(lab[:,:,0])
#
#d = cv2.imread("C:/Users/admin.H115/git/image-data/test_shadow/003.jpg")
#lab2 = cv2.cvtColor(d,cv2.COLOR_BGR2LAB)
#print(lab2[:,:,2])

gamma = 0.7
lookup_table = np.empty((1,256), np.uint8)
for i in range(256):
    lookup_table[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
 
# Look up tableを使って画像の輝度値を変更
imga = cv2.LUT(b, lookup_table)
cv2.namedWindow('imagea', cv2.WINDOW_AUTOSIZE)
cv2.imshow('imagea', b)
cv2.imshow('gammma', imga)
cv2.waitKey()
