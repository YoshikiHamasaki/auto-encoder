from my_module import image_rename.py as rename
from my_module import image_process.py as process
from my_module import make_csv.py as make_csv

rename.rename("../image-data/trimming/")
process.Binariz("../image-data/trimming/","../image-data/")#$B%H%j%_%s%08e#2CM(B
make_csv.make_csv("../image-data/","/csv/.csv",0)#$B#2CM8e(B


