from haikan import data_load



num_epochs = 20                   
learning_rate = 0.001             
out_dir = "result"                

data_load("train.csv","test_bad.csv"
        ,"../image-data/AE_train_bin/","AE_test_bad_bin/")


