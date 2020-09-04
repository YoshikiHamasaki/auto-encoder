import cv2
import glob

image_number = 16
def Binariz(input_path, output_path):
    files = glob.glob(input_path + '/*')
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
        cv2.imwrite(output_path + "{0:03d}.jpg".format(i),img)
        
