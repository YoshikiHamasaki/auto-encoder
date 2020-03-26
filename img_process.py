import cv2


image_number = 7


for i in range(image_number):
    img = cv2.imread("../image-data/dirt/" + "{0:06d}.jpg".format(i), cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    cv2.imwrite("../image-data/aiu/" + "{0:06d}.jpg".format(i),img)

    
img2 = cv2.imread("../image-data/aiu/" + "{0:06d}.jpg".format(1))  
print(img2)  
cv2.imshow("a",img2)
cv2.waitKey()
