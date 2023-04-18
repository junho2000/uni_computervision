import cv2

img = cv2.imread('../data/Lena.png')

resized_img = cv2.resize(img,(128,128))

cv2.namedWindow("resized", cv2.WINDOW_AUTOSIZE)
cv2.imshow("resized", resized_img)
cv2.waitKey()
cv2.destroyAllWindows()