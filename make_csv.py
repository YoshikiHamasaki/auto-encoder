import csv
import glob
import numpy as np
import os

y = []
path = 'AE_test_bad/'
files = []
for filename in os.listdir(path):
     if os.path.isfile(os.path.join(path, filename)): #ファイルのみ取得
         files.append(filename)

with open('test_bad.csv', 'w',newline="") as f: 
   writer = csv.writer(f, lineterminator="\n")
   writer.writerows([["ImageName","ImageLabel"]])
   for data in files:
       writer.writerows([[data, 1]]) 
       

print(files[0])
