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

test_path = "C:/Users/admin.H115/git/image-data/train_shadow_bright_about100" 
csv_name = "csv/lab data/train_shadow_bright_about100.csv"

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
    df.to_csv(csv_name, mode ="a",header = None)

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
