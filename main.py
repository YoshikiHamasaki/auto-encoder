from my_module import start_detection as detect
from my_module import classify_mix_image as classify
import glob
import os
from natsort import natsorted

COLOR = 0b1
BINARY = 0b10
BRIGHTNESS = 0b100
ON = 0b1000
global image_type 
image_type = 0
MIX = 0

class Filepath:
    def __init__(self,image_folder,csv_folder):
        self.image_path = os.path.join("../image-data/",image_folder)
        self.csv_path = os.path.join("csv/label_name_data",csv_folder,csv_folder + ".csv")
    
    def image(self):
        return self.image_path

    def csv(self):
        return self.csv_path

    def for_mix_image(self):
        self.classify_image_path = natsorted(glob.glob(os.path.join("../image-data/",image_folder,"classify/*")))
        return self.classify_image_path

    def for_mix_csv(self):
        self.classify_csv_path = natsorted(glob.glob(os.path.join("csv/label_name_data",csv_folder + "/*")))
        return self.classify_csv_path

    def for_train_shadow(self):




    


######### setting parameter #################
#image_type |= BRIGHTNESS#
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
test_img_index = 1
train_name = ""
#############################################
        
##### for mix param #####
MIX |= ON
mix_folder_name = "test_mix"
mix_test_image = natsorted(glob.glob(os.path.join("../image-data/",mix_folder_name,"classify/*")))
mix_test_csv = natsorted(glob.glob(os.path.join("csv/label_name_data",mix_folder_name + "/*")))
folder_name1_epoch = 50
folder_name2_epoch = 50
folder_name3_epoch = 50
folder_name4_epoch = 100
#########################

train_name = os.path.basename(train_image_path)

def main():
    global image_type

    if MIX & ON:
        classify.classify("../image-data/test_mix" ,"csv/analysis/test_mix_param.csv")
        for i in range(len(mix_test_image)):
            if os.path.basename(mix_test_image[i]) == "about40":
                train_csv_path = "csv/label_name_data/train_shadow/train_shadow_bright_about40.csv"
                test_csv_path =  mix_test_csv[i]
                train_name = "train_shadow_bright_about40"
                train_image_path  = "../image-data/train_shadow_bright_about40" 
                test_image_path = mix_test_image[i]
                image_type |= BRIGHTNESS
                print("40")

            if os.path.basename(mix_test_image[i]) == "about60":
                train_csv_path = "csv/label_name_data/train_shadow/train_shadow_bright_about60.csv"
                test_csv_path =  mix_test_csv[i]
                train_name = "train_shadow_bright_about60"
                train_image_path = "../image-data/train_shadow_bright_about60" 
                test_image_path = mix_test_image[i]
                image_type |= BRIGHTNESS
                print("60")


    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,3*28*28,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        
    if image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    if image_type & BRIGHTNESS:
        detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        print("a")

    else:
        print("select image_type")



                


if __name__ == "__main__":
    main()
