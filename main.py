from my_module import start_detection as detect
import glob
import os

COLOR = 0b1
BINARY = 0b10
BRIGHTNESS = 0b100
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
    else:
        print("select image_type")

if __name__ == "__main__":
    main()
