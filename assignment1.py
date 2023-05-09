import cv2
import numpy as np



img = cv2.imread("/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A1/img_example.JPG")
rows,cols = img.shape[0:2]

# ---① 라디안 각도 계산(60진법을 호도법으로 변경)
ang = np.pi/3

# ---② 회전을 위한 변환 행렬 생성
m601 = np.float32( [[ np.cos(ang), -1* np.sin(ang), rows],
                    [np.sin(ang), np.cos(ang), 0]])
m602 = cv2.getRotationMatrix2D((0,0),-60,1) 

# ---③ 회전 변환 행렬 적용
r601 = cv2.warpAffine(img,m601,(rows,cols))
r602 = cv2.warpAffine(img,m602,(rows,cols))

# ---④ 결과 출력
cv2.imshow("origin", img)
cv2.imshow("601", r601)
cv2.imshow("602", r602)
cv2.waitKey()
cv2.destroyAllWindows()