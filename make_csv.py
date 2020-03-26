import csv
import glob
import numpy as np
import os
from natsort import natsorted

y = []

#csvを作成したいフォルダ入力
path = '../image-data/dirt/'
files = []
for filename in os.listdir(path):
     if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
         files.append(filename)

with open('aaaa.csv', 'w',newline="") as f: 
   writer = csv.writer(f, lineterminator="\n")
   writer.writerows([["ImageName","ImageLabel"]])
   for data in natsorted(files):
      writer.writerows([[data, 0]]) 
