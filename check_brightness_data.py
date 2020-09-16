import pandas as pd
import os
import cv2
from my_module import image_rename as rename


test_path = "C:/Users/admin.H115/git/image-data/test_shadow_good" 
csv_name = "csv/lab data/test_shadow_good_data.csv"
rename.rename(test_path)

img_name = os.listdir(test_path) 

for i in range(len(img_name)):

    image = cv2.imread(os.path.join(test_path, img_name[i]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    image_lab = image[:,:,0]
    #image_lab = image_lab.flatten()
#    with open("csv/train_lab_data.csv", "a", newline = "") as f:
#        writer = csv.writer(f, lineterminator = "\n")
#        writer.writerows([[img_name[i],image_lab]])

    df = pd.DataFrame(image_lab)
    df["num"] = img_name[i]
    df = df.set_index("num")
    df.to_csv(csv_name, mode ="a",header = None)
