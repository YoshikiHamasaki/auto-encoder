from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
from my_module import classify_mix_image as classify
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd



def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img/2 +0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#img_name = os.listdir(test_path) 

#for i in range(len(img_name)):
#
#    image = cv2.imread(os.path.join(test_path, img_name[i]))
#    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
#    image_lab = image[:,:,0]
#    #image_lab = image_lab.flatten()
##    with open("csv/train_lab_data.csv", "a", newline = "") as f:
##        writer = csv.writer(f, lineterminator = "\n")
##        writer.writerows([[img_name[i],image_lab]])
#
#    df = pd.DataFrame(image_lab)
#    df["num"] = img_name[i]
#    df = df.set_index("num")
##    df.to_csv(csv_name, mode ="a",header = None)

classify.classify("../image-data/test_mix" ,"csv/analysis/test_mix_param.csv")
mix_test_image = natsorted(glob.glob(os.path.join("../image-data/test_mix","classify/*")))

class A:
    def __init__(self):
        self.num = 0

    def test(self,txit):
        self.txit = natsorted(glob.glob(os.path.join("../image-data/",txit,"classify/*")))
        return(self.txit)

aiu = A()
print(aiu.test("test_mix"))
print(aiu.test)
mix_folder_name = "test_mix"
print(mix_test_csv)
