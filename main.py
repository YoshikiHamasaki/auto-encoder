from my_module import start_detection as detect
import glob
import os

COLOR = 0b1
BINARY = 0b10
image_type = 0

######### setting parameter #################
image_type |= COLOR
AE_type = "COLOR"
num_epochs = 150
optimizer_type = "SGD"
learning_rate = 0.05
weight_decay = 1e-5
input_size = 3*28*28
train_csv_path = "csv/8_12_train.csv" 
test_csv_path = "csv/test_good.csv" 
train_image_path = "../image-data/8_12_train" 
test_image_path = "../image-data/test_good" 
test_img_index = "ALL"
#############################################
        

train_name = os.path.basename(train_image_path)

def main():

    if image_type & COLOR:
        detect.for_color_detect(AE_type,num_epochs,optimizer_type,learning_rate,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,test_img_index,train_name)
        
    elif image_type & BINARY:
        detect.for_bin_detect(AE_type,num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
                train_csv_path,test_csv_path,train_image_path,test_image_path,train_name)
    
    else:
        print("select image_type")

if __name__ == "__main__":
    main()
