from my_module import image_rename as rename
from my_module import image_process as process
from my_module import make_csv as make_csv

rename.rename("../image-data/8_12_train/")
#process.Binariz("../image-data/trimming/","../image-data/")#トリミング後２値
make_csv.make_csv("../image-data/8_12_train","csv/8_12_train.csv",0)#２値後


