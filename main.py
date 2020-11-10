import os
import pandas as pd
from my_module import start_detection as detect
from my_module.classify_mix_image import classify
from my_module.FilePath_Class import FilePath
from my_module.get_dir_size import get_dir_size


train_path_box = FilePath("8_12_train","8_12_train")
#影の学習用画像フォルダ名
train_shadow_path_box = FilePath("train_shadow","train_shadow")
#テストしたいフォルダ名を入力
test_mix_path_box = FilePath("test_11_6","test_11_6")


#train_name = os.path.basename(train_image_path)

def main():

    COLOR = 0b1
    BINARY = 0b10
    BRIGHTNESS = 0b100
    ON = 0b1000
    image_type = 0
    MIX = 0
    ######### setting parameter #################

    #image_type |= BRIGHTNESS
    AE_type = "LAB"
    num_epochs = 100
    optimizer_type = "SGD"
    learning_rate = 0.05
    weight_decay = 1e-5
    input_size = 28*28
    train_csv_path = "csv/label_name_data/train_shadow/train_shadow_bright_about60.csv" 
    test_csv_path = "csv/label_name_data/test_good.csv" 
    train_image_path = "../image-data/train_shadow_bright_about60" 
    test_image_path = "../image-data/test_good" 
    test_img_index = "ALL"
    train_name = ""
    #############################################

    ##### for mix param #####
    MIX |= ON
    folder_name1_epoch = 50
    folder_name2_epoch = 50
    folder_name3_epoch = 50
    folder_name4_epoch = 350
    #########################

    train_path_box = FilePath("8_12_train","8_12_train")
    train_shadow_path_box = FilePath("train_shadow","train_shadow")
    test_mix_path_box = FilePath("test_11_6","test_11_6")
    #train_name = os.path.basename(train_image_path)



    judge_result = pd.DataFrame()
    judge_result.columns = ["E_list","judge_list","train_model_list","index_list"]
    judge_result.to_csv("csv/result.csv")



    if MIX & ON:
        classify(test_mix_path_box.image() ,"csv/analysis/test_11_6_param.csv")
        for i in range(len(test_mix_path_box.for_mix_image())):
            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about40" and get_dir_size(test_mix_path_box.for_mix_image()[i])!= 0:
                train_name = "train_shadow_bright_about40"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("40"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("40"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "40")
                print("about40 end")


            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about60" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about60"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("60"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("60"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "60")
                print("about60 end")



            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about100" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about100"
                detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_shadow_path_box.for_train_shadow_csv("100"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("100"),test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "100")
                print("about100 end")
                


            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "over135" and get_dir_size(test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "8_12_train"
                detect.for_color_detect("COLOR",folder_name4_epoch,optimizer_type,learning_rate,3*28*28,
                train_path_box.csv(),test_mix_path_box.for_mix_csv()[i],train_path_box.image(),test_mix_path_box.for_mix_image()[i],
                test_img_index,train_name,error_th = "color")
                print("over135 end")



    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,3*28*28,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        
    if image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    if image_type & BRIGHTNESS:
        detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)

    else:
        print("miss! select image_type")


if __name__ == "__main__":
    main()
