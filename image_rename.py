import os
import cv2
import glob

path = 'AE_train/'
files = glob.glob(path + '/*')
for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:03d}.jpg'.format(i)))
