import pandas as pd
import os
import cv2
import numpy as np

test_path = "C:/Users/admin.H115/git/image-data/train_shadow_bright_about100" 
csv_name = "csv/analysis/train_shadow_bright_about100_param.csv"

img_name = os.listdir(test_path) 


abusolute_dif = []
ave = []
for i in range(1):

    image = cv2.imread(os.path.join(test_path, img_name[i]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image_lab = image[:,:,0]
    abusolute_dif.append(np.max(image_lab)-np.min(image_lab))
    ave.append
    print(np.min(image_lab))


#df = pd.DataFrame()
    #df = df.set_index("num")
    #df.to_csv(csv_name, mode ="a",header = None)


