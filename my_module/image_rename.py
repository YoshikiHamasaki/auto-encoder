import os
import cv2
import glob

def rename(path):
    files = glob.glob(path + '/*')
    for i, f in enumerate(files):
        os.rename(f, os.path.join(path, '{0:06d}.jpg'.format(i)))
