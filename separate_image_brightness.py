import pandas as pd
import os
import cv2
import numpy as np

test_path = "C:/Users/admin.H115/git/image-data/train_shadow_bright_about100" 
csv_name = "csv/analusis/train_shadow_bright_about100_param.csv"

img_name = os.listdir(test_path) 


for i in range(1):

    image = cv2.imread(os.path.join(test_path, img_name[i]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image_lab = image[:,:,0]
    image_lab.reshape(1,28*28)
    print(image_lab.shape)
    print(np.min(image_lab))
    df = pd.DataFrame()
    #df = df.set_index("num")
    #df.to_csv(csv_name, mode ="a",header = None)


