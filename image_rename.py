import os
import cv2
import glob
from natsort import natsorted

path = '../image-data/dirt/'
files = sorted(glob.glob(path + '/*'))
for i, f in enumerate(files):
    os.rename(f, os.path.join(path, '{0:03d}.jpg'.format(i)))
