import pandas as pd
import os
import cv2
import numpy as np
import scipy.stats as stats
import math

test_path = "C:/Users/admin.H115/git/image-data/test_shadow_bad" 
csv_name = "csv/analysis/test_shadow_param.csv"

img_name = os.listdir(test_path) 


abusolute_dif = []
ave = []
mode = []
list_data = []


for i in range(len(img_name)):

    image = cv2.imread(os.path.join(test_path, img_name[i]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image_lab = image[:,:,0]
    abusolute_dif.append(np.max(image_lab)-np.min(image_lab))
    ave.append(np.average(image_lab))
    uniqs,counts = np.unique(image_lab,return_counts=True)
    mode.append(str(uniqs[counts == np.amax(counts)]).strip("[]"))

list_data = [abusolute_dif,
            ave,
            mode]


df = pd.DataFrame(data = list_data)
df_T = df.T
df_T.columns = ["abusolute_diff","average","mode"]
df_T.to_csv(csv_name, mode ="w")


