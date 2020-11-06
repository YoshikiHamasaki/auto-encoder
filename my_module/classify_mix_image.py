import os
import pandas as pd
from natsort import natsorted
from my_module import make_csv 
from my_module import image_brightness_search as search
from my_module import image_rename 
import shutil


#テストしたいフォルダと，それのparamのcsv必要
folder_name1 = "about40"
folder_name2 = "about60"
folder_name3 = "about100"
folder_name4 = "over135"



list_folder = []
def classify(test_path,csv_path):

    check_path = test_path + "/" + "classify"

    mycheck = os.path.isdir(check_path)
    
    folder_list = [folder_name1,folder_name2,folder_name3,folder_name4]

    test_file_name = os.path.basename(test_path)
    

    if not mycheck:

        search.search(test_path,csv_path)
        os.mkdir(check_path)
        for i in folder_list:
            os.mkdir(check_path + "/" + i)

        #os.mkdir(os.path.join(check_path,"csv"))

        df = pd.read_csv(csv_path)


        image_name = df["image_name"]
        folder = df["class_ave_folder"]

        for x,y in enumerate(folder):

            if y == folder_name1:
                list_folder.append(y) 
                shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, folder_name1 ,str(image_name[x])))
                image_rename.rename(os.path.join(check_path, folder_name1))

            elif y == folder_name2:
                list_folder.append(y) 
                shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, folder_name2 ,str(image_name[x])))
                image_rename.rename(os.path.join(check_path, folder_name2))

            elif y == folder_name3:
                list_folder.append(y) 
                shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, folder_name3 ,str(image_name[x])))
                image_rename.rename(os.path.join(check_path, folder_name3))

            elif y == folder_name4:
                list_folder.append(y) 
                shutil.copy(os.path.join(test_path,image_name[x]),os.path.join(check_path, folder_name4 ,str(image_name[x])))
                image_rename.rename(os.path.join(check_path, folder_name4 ))
        
    if not os.path.isdir(os.path.join("C:/Users/admin.H115/git/auto-encoder/csv/label_name_data",test_file_name)):
        os.mkdir(os.path.join("C:/Users/admin.H115/git/auto-encoder/csv/label_name_data",test_file_name))

    for i,j in enumerate(folder_list):
        make_csv.make_csv(os.path.join(check_path,folder_list[i]),os.path.join("C:/Users/admin.H115/git/auto-encoder/csv/label_name_data", test_file_name ,folder_list[i] + "_data.csv"),0)
