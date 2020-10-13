from my_module import start_detection as detect
import glob
import os
from natsort import natsorted

COLOR = 0b1
BINARY = 0b10
BRIGHTNESS = 0b100
ON = 0b1000
image_type = 0
MIX = 0

class 
######### setting parameter #################
image_type |= BRIGHTNESS
AE_type = "LAB"
num_epochs = 100
optimizer_type = "SGD"
learning_rate = 0.05
weight_decay = 1e-5
input_size = 28*28
train_csv_path = "csv/train_shadow_bright_about60.csv" 
test_csv_path = "csv/test_shadow_bad.csv" 
train_image_path = "../image-data/train_shadow_bright_about60" 
test_image_path = "../image-data/test_shadow_bad" 
test_img_index = 2
train_name = ""
#############################################
        
##### for mix param #####
MIX |= ON
mix_folder_name = "test_mix"
mix_test_image = natsort(glob.glob(os.path.join("../image-data/",mix_folder_name,"classify/*")))
folder_name1_epoch = 50
folder_name2_epoch = 50
folder_name3_epoch = 50
folder_name4_epoch = 100
#########################

train_name = os.path.basename(train_image_path)

def main():



    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,input_size = 3*28*28,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        
    elif image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    elif image_type & BRIGHTNESS:
        detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)

    else:
        print("select image_type")


    if MIX & ON:
        classify.classify("../image-data/test_mix" ,"csv/analysis/test_mix_param.csv")
        for i in mix_test_image:
            if os.basename(i) == "about40":
                train_csv_path = 
                


if __name__ == "__main__":
    main()
