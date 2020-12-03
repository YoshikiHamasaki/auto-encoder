import os
import sys
import pandas as pd
from my_module import start_detection as detect
from my_module.classify_mix_image import classify
from my_module.FilePath_Class import FilePath
from my_module.get_dir_size import get_dir_size


#train_name = os.path.basename(train_image_path)

def main():

    OFF = 0
    COLOR = 0b1
    BINARY = 0b10
    BRIGHTNESS = 0b100
    ON = 0b1000
    image_type = 0
    MIX = 0


    #colorの学習用フォルダ
    train_path_box = FilePath("8_12_train","8_12_train") 
     #影の学習用画像フォルダ名
    train_shadow_path_box = FilePath("train_shadow","train_shadow")
     #テストしたいフォルダ名を入力
    test_mix_path_box = FilePath("test_11_11","test_11_11")
    #train_name = os.path.basename(train_image_path)


    ############# setting parameter #################
    image_type |= 0  # COLOR or BRIGHTNESS
    AE_type = "COLOR"
    num_epochs = 150
    optimizer_type = "SGD"
    learning_rate = 0.05
    weight_decay = 1e-5
    input_size = 3*28*28
    train_csv_path = "csv/label_name_data/color_train_retry.csv" 
    test_csv_path = "csv/label_name_data/test_11_6/over135_data.csv" 
    train_image_path = "../image-data/color_train_retry" 
    test_image_path = "../image-data/test_11_6/classify/over135" 
    test_img_index = "ALL"
    train_name = "color_train_retry"
    #############################################


    ##### for mix param #####
    MIX |= ON
    folder_name1_epoch = 100
    folder_name2_epoch = 101
    folder_name3_epoch = 100
    folder_name4_epoch = 100
    folder_name5_epoch = 340
    #########################



    column = ["E_list","judge_list","train_model_list","index_list"]
    judge_result = pd.DataFrame(data = column)
    judge_result = judge_result.T
    judge_result.to_csv("csv/result.csv",mode = "w",index = False,header = False)


    if MIX & ON:
        classify(test_mix_path_box.image() ,test_mix_path_box.for_param_csv())
        for i in range(len(test_mix_path_box.for_mix_image())):
            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about40" and get_dir_size(
                    test_mix_path_box.for_mix_image()[i])!= 0:
                train_name = "train_shadow_bright_about40"
                detect.for_expert_detect("LAB",folder_name1_epoch,optimizer_type,
                        learning_rate,weight_decay,28*28,
                train_shadow_path_box.for_train_shadow_csv("40"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("40"),
                test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "40")
                print("about40 end")


            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about60" and get_dir_size(
                    test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about60"
                detect.for_expert_detect("LAB",folder_name2_epoch,optimizer_type,
                        learning_rate,weight_decay,28*28,
                train_shadow_path_box.for_train_shadow_csv("60"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("60"),
                test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "60")
                print("about60 end")



            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about100" and get_dir_size(
                    test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about100"
                detect.for_expert_detect("LAB",folder_name3_epoch,optimizer_type,
                        learning_rate,weight_decay,28*28,
                train_shadow_path_box.for_train_shadow_csv("100"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("100"),
                test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "100")
                print("about100 end")
                


            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "about120" and get_dir_size(
                    test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "train_shadow_bright_about120"
                detect.for_expert_detect("LAB",folder_name4_epoch,optimizer_type,
                        learning_rate,weight_decay,28*28,
                train_shadow_path_box.for_train_shadow_csv("120"),test_mix_path_box.for_mix_csv()[i],
                train_shadow_path_box.for_train_shadow_image("120"),
                test_mix_path_box.for_mix_image()[i],test_img_index,train_name,error_th = "120")
                print("about120 end")



            if os.path.basename(test_mix_path_box.for_mix_image()[i]) == "over135" and get_dir_size(
                    test_mix_path_box.for_mix_image()[i]) != 0:
                train_name = "8_12_train"
                detect.for_expert_detect("COLOR",folder_name5_epoch,optimizer_type,
                        learning_rate,weight_decay,3*28*28,
                train_path_box.csv(),test_mix_path_box.for_mix_csv()[i],
                train_path_box.image(),test_mix_path_box.for_mix_image()[i],
                test_img_index,train_name,error_th = "color")
                print("over135 end")

        sys.exit()



    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,3*28*28,
        train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,
        train_name,error_th = "color")
        print("color end") 
    
    if image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,
        input_size,train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    if image_type & BRIGHTNESS:
        detect.for_brightness_detect(AE_type,num_epochs,optimizer_type,learning_rate,
        weight_decay,input_size,train_csv_path,test_csv_path,train_image_path,
        test_image_path,test_img_index,train_name,error_th = "100")
        print("lab end")

    if image_type & 0:
        print("miss! select image_type")


if __name__ == "__main__":
    main()
