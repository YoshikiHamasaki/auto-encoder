from my_module import start_detection as detect
import glob
import os

COLOR = 0b1
BINARY = 0b10
BRIGHTNESS = 0b100
MIX = 0b1000
image_type = 0

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
#############################################
        
##### for mix param #####
image_type |= MIX
train
folder_name1_epoch = 50
folder_name2_epoch = 50
folder_name3_epoch = 50
folder_name4_epoch = 100
#########################

train_name = os.path.basename(train_image_path)

def main():



    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        
    elif image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    elif image_type & BRIGHTNESS:
        detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)

    elif image_type & MIX:
        classify.classify("../image-data/test_shadow_bad" ,"csv/analysis/test_bad_param.csv")
        detect.for_brightness_detect

    A
    else:
        print("select image_type")

if __name__ == "__main__":
    main()
