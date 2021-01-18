import math
import cv2
import numpy as np
import os
import glob
import pandas as pd
from natsort import natsorted


img_name_list = natsorted(glob.glob("../image-data/test_12_15_1_remove/classify/**/*.jpg",recursive = True))

img2 = cv2.imread("../image-data/test_12_8_2/classify/about100/000045.jpg")
img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
img2 = cv2.resize(img2,None,fx = 5,fy =5)
cv2.imshow("a",img2)
cv2.waitKey(0)

entropy_list = []
entropy_list2 = []

column = ["img_name_list","entropy_list","entropy_list2"]
entropy_result = pd.DataFrame(data = column)
entropy_result = entropy_result.T
entropy_result.to_csv("csv/entropy_result.csv",mode = "w",index = False,header = False)

for i in range(len(img_name_list)):

    img = cv2.imread(img_name_list[i])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    histgram = [0]*256
    
    for y in range(28):
        for x in range(28):
            histgram[gray[y,x]] += 1
    
    size = img.shape[0]*img.shape[1]
    entropy = 0
    entropy2 = 0
    
    for i in range(256):
        p = (histgram[i]/size)
        p2 = (histgram[i]/size)*10
        
        if p == 0:
            continue
        entropy -= p*math.log2(p)
        entropy2 -= p*math.log2(p2)
    
    entropy_list.append(entropy)
    entropy_list2.append(entropy2*10)


all_data = [entropy_list,entropy_list2,img_name_list]
df = pd.DataFrame(data = all_data)
df = df.T
df.to_csv("csv/entropy_result.csv",mode = "a",header = False,index = False)


