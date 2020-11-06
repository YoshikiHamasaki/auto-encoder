import csv
import glob
import numpy as np
import os
from natsort import natsorted

def make_csv(path, output_file, label):
    #csvを作成したいフォルダ入力
    files = []
    for filename in os.listdir(path):
         if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
             files.append(filename)
    with open(output_file, 'w',newline="") as f: 
       writer = csv.writer(f, lineterminator="\n")
       writer.writerows([["ImageName","ImageLabel"]])
       for data in natsorted(files):
           writer.writerows([[data, label]]) 
#make_csv("D:/image/slack/base/", 'D:/image/slack/base/test.csv', 0)
