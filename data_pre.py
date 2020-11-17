from my_module import image_rename as rename
from my_module import image_process as process
from my_module import make_csv as make_csv

rename.rename("../image-data/11_13_search")

#process.Binariz("../image-data/trimming/","../image-data/")#トリミング後２値
make_csv.make_csv("../image-data/11_13_search","csv/label_name_data/11_13_search",0)
