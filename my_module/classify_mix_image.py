import os
import pandas as pd
from my_module import image_brightness_search as search
import shutil



list_folder = []
def classify(test_path,csv_path):

    check_path = test_path + "/" + "classify"

    mycheck = os.path.isdir(check_path)

    if not mycheck:

        search.search(test_path,csv_path)
        os.mkdir(check_path)
        folder_list = ["about40","about60","about100","over135"]
        for i in folder_list:
            os.mkdir(check_path + "/" + i)


    df = pd.read_csv(csv_path)
    image_name = df["image_name"]
    folder = df["class_ave_folder"]

    for x,y in enumerate(folder):

        if y == "about40":
            list_folder.append(y) 
            shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, "about40" ,str(image_name[x])))
        elif y == "about60":
            list_folder.append(y) 
            shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, "about60" ,str(image_name[x])))
        elif y == "about100":
            list_folder.append(y) 
            shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, "about100" ,str(image_name[x])))
        elif y == "over135":
            list_folder.append(y) 
            shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, "over135" ,str(image_name[x])))
