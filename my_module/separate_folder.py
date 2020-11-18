import pandas as pd
import os
import cv2
import numpy as np
import scipy.stats as stats
import math
from natsort import natsorted


def separate(test_path,csv_path):

    img_name = natsorted(os.listdir(test_path))
    
    
    image_name = []
    abusolute_dif = []
    ave = []
    mode = []
    class_ave_folder =[]
    list_data = []
    
    
    for i in range(len(img_name)):
    
        image = cv2.imread(os.path.join(test_path, img_name[i]))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        image_lab = image[:,:,0]

        image_name.append(img_name[i])
        abusolute_dif.append(np.max(image_lab)-np.min(image_lab))
        ave.append(np.average(image_lab))
        uniqs,counts = np.unique(image_lab,return_counts=True)
        mode.append(str((uniqs[counts == np.amax(counts)].min())).strip("[]"))

        if np.average(image_lab) <= 49:
            class_ave_folder.append("about40")
        
        elif 49 < np.average(image_lab) <= 79:
            class_ave_folder.append("about60")

        elif 79 < np.average(image_lab) <= 105:
            class_ave_folder.append("about100")

        elif 105 < np.average(image_lab) <= 135: 
            class_ave_folder.append("about120")

        elif 135 < np.average(image_lab):
            class_ave_folder.append("over135")
     
        else:
            print("else")
    
    list_data = [image_name,abusolute_dif,ave,mode,class_ave_folder]
    
    df = pd.DataFrame(data = list_data)
    df_T = df.T
    df_T.columns = ["image_name","abusolute_diff","average","mode","class_ave_folder"]
    df_T.to_csv(csv_path, mode ="w",index = False)
