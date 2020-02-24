import cv2


image_number = 16


for i in range(image_number):
    img = cv2.imread("../image-data/AE_test_good/" + "{0:03d}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite("../image-data/AE_test_good_bin/" + "{0:03d}.jpg".format(i),img)
    
