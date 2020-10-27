from my_module import start_detection as detect
from my_module import classify_mix_image as classify
import glob
import os
from natsort import natsorted


class Filepath():
    def __init__(self, image_folder, csv_folder):
        self.image_folder = image_folder
        self.csv_folder = csv_folder
        self.image_path = os.path.join("../image-data/",self.image_folder)
        self.csv_path = os.path.join("csv/label_name_data",self.csv_folder,self.csv_folder + ".csv")

    def image(self):
        return self.image_path

    def csv(self):
        return self.csv_path

    def for_mix_image(self):
        self.classify_image_path = natsorted(glob.glob(os.path.join("../image-data",self.image_folder,"classify/*")))
        return self.classify_image_path

    def for_mix_csv(self):
        self.classify_csv_path = natsorted(glob.glob(os.path.join("csv/label_name_data",self.csv_folder + "/*")))
        return self.classify_csv_path

    def for_train_shadow_image(self,num):
        self.train_shadow_image_path = os.path.join("../image-data",self.image_folder + "_bright_about" + num)
        return self.train_shadow_image_path

    def for_train_shadow_csv(self,num):
        self.train_shadow_csv_path = os.path.join("csv/label_name_data/train_shadow" ,self.csv_folder + "_bright_about" + num + ".csv")
        return self.train_shadow_csv_path 



def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


train_path_box = Filepath("8_12_train","8_12_train")
train_shadow_path_box = Filepath("train_shadow","train_shadow")
test_mix_path_box = Filepath("test_mix","test_mix")


#train_name = os.path.basename(train_image_path)


    COLOR = 0b1
    BINARY = 0b10
    BRIGHTNESS = 0b100
    ON = 0b1000
    image_type = 0
    MIX = 0
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
    folder_name1_epoch = 50
    folder_name2_epoch = 50
    folder_name3_epoch = 50
    folder_name4_epoch = 100
    #########################
    if MIX & ON:
        classify.classify(test_mix_path_box.image() ,"csv/analysis/test_mix_param.csv")
        for i in range(len(test_mix_path_box.for_mix_image())):
            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about40" and get_dir_size(test_mix_path_box.for_mix_image()[i])!= 0:
                train_name = "train_shadow_bright_about40"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("40"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("40"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name)



            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about60" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about60"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("60"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("60"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name)




            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about100" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about100"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("100"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("100"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name)
                


            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "over135" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "8_12_train"
                num_epochs = folder_name4_epoch
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_path_box.csv(),test_mix_path_box.for_mix_csv()[i],train_path_box.image(),test_mix_path_box.for_mix_image()[i],
                test_img_index,train_name)



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
