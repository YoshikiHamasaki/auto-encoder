import os

def classify(test_image_path):

    check_path = test_image_path + "/" + "classify"

    mycheck = os.path.isdir(check_path)

    if not mycheck:   
        os.mkdir(check_path)
        folder_list = ["about40","about60","about100","over135"]
        for i in folder_list:
            os.mkdir(check_path + "/" + i)

        
