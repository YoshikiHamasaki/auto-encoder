from my_module import image_rename as rename
from my_module import image_process as process
from my_module import make_csv as make_csv

rename.rename("../image-data/train_shadow/")

#process.Binariz("../image-data/trimming/","../image-data/")#$B%H%j%_%s%08e#2CM(B
make_csv.make_csv("../image-data/train_shadow","csv/train_shadow.csv",0)#$B#2CM8e(B
