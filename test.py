from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

test_path = "C:/Users/admin.H115/git/image-data/test_shadow_bad/" 
test_path2 ="C:/Users/admin.h115/git/image-data/test_shadow/194.jpg"


#train_loader1, test_loader1 = dataset.lab_make_data_set("csv/train_shadow.csv","csv/test_shadow_bad.csv","../image-data/train_shadow","../image-data/test_shadow_bad")


train_loader, test_loader = dataset.color_make_data_set("csv/train.csv","csv/test_bad.csv","../image-data/train","../image-data/test_bad")

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

img_name = os.listdir(test_path) 

for i in range(len(img_name)):

    image = cv2.imread(os.path.join(test_path, img_name[i]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image_lab = image[:,:,0]
    #image_lab = image_lab.flatten()
#    with open("csv/train_lab_data.csv", "a", newline = "") as f:
#        writer = csv.writer(f, lineterminator = "\n")
#        writer.writerows([[img_name[i],image_lab]])

    df = pd.DataFrame(image_lab)
    df["num"] = img_name[i]
    df = df.set_index("num")
    df.to_csv("csv/lab data/test2.csv", mode ="a",header = None)
#print(df)
#iterator = iter(train_loader1)
#img, _ = next(iterator)
#img = (img/2 +0.5)*255
#print(img.shape)
#print(img[9])
#plt.imshow(npimg[2])
#plt.show()

#a = cv2.imread("C:/Users/admin.H115/git/image-data/8_12_train/166.jpg")
#image = cv2.cvtColor(a,cv2.COLOR_BGR2HSV_FULL)
#
#b = cv2.imread(test_path)
#hsv1 = cv2.cvtColor(b,cv2.COLOR_BGR2HSV_FULL)
#lab = cv2.cvtColor(b,cv2.COLOR_BGR2LAB)
#b = cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
#
#
#plt.figure(figsize=(9,7))
#plt.subplot(1,3,1)
#plt.imshow(b)
#plt.subplot(1,3,2)
#plt.imshow(hsv1)
#plt.subplot(1,3,3)
#plt.imshow(lab)
#plt.show()



#cv2.namedWindow('imagea', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('imagea', b)
#cv2.imshow('gammma', imga)
#cv2.waitKey()
