from my_module import image_rename.py as rename
from my_module import image_process.py as process
from my_module import make_csv.py as make_csv

rename.rename("../image-data/trimming/")
process.Binariz("../image-data/trimming/","../image-data/")#トリミング後２値
make_csv.make_csv("../image-data/","/csv/.csv",0)#２値後


