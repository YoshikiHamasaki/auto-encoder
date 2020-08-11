from my_module import start_detection as detect

COLOR = 0b1
BINARY = 0b10
image_type = 0

#### setting parameter ######################
image_type |= COLOR
num_epochs = 100
optimizer_type = "SGD"
learning_rate = 0.05
weight_decay = 1e-5
input_size = 3*28*28
train_csv_path = "csv/train.csv" 
test_csv_path = "csv/test_good.csv" 
train_image_path = "../image-data/AE_train" 
test_image_path = "../image-data/AE_test_good" 
#############################################
        
if image_type & COLOR:
    detect.for_color_detect(num_epochs,optimizer_type,learning_rate,input_size,
            train_csv_path,test_csv_path,train_image_path,test_image_path)

elif image_type & BINARY:
    detect.for_bin_detect(num_epochs,optimizer_type,learning_rate,weight_decay,input_size,
            train_csv_path,test_csv_path,train_image_path,test_image_path)

else:
    print("select image_type")
