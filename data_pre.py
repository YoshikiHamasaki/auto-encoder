from my_module import image_rename as rename
from my_module import image_process as process
from my_module import make_csv as make_csv

#rename.rename("../image-data/test_11_6")

#process.Binariz("../image-data/trimming/","../image-data/")#トリミング後２値
make_csv.make_csv("../image-data/test_11_6","csv/label_name_data/test_11_6.csv",0)#２値後
