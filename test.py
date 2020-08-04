from natsort import natsorted
import glob
import cv2
from my_module import make_data_set as dataset
import numpy as np

train_loader, test_loader = dataset.make_data_set("csv/train.csv","csv/test_good.csv","../image-data/AE_train","../image-data/AE_test_good_bin")


iterator = iter(train_loader)
img, _ = next(iterator)
img_new = img.reshape((3,28,28).transpose(1,2,0))
plt.imshow(img_new[0])
plt.show()
