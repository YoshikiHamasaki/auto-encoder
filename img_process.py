import cv2

for i in range(208):
    img = cv2.imread("AE_test_bad/" + "{0:03d}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("otsu", img)
    cv2.imwrite("AE_test_bad_bin/" + "{0:03d}.jpg".format(i),img)
    
