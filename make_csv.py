import csv
import glob
import numpy as np
import os

y = []

#csvを作成したいフォルダ入力
path = '../image-data/AE_test_good_bin/'
files = []
for filename in os.listdir(path):
     if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
         files.append(filename)

with open('test_good.csv', 'w',newline="") as f: 
   writer = csv.writer(f, lineterminator="\n")
   writer.writerows([["ImageName","ImageLabel"]])
   for data in files:
       writer.writerows([[data, 0]]) 
