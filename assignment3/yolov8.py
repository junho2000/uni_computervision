import cv2
from ultralytics import YOLO
import time

img_pth = "/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A3/img_example.JPG"

model = YOLO("yolov8n.pt")
 
start_time = time.time()
results = model(source=img_pth)
end_time = time.time()
execution_time = end_time - start_time

res_plotted = results[0].plot()

cv2.imshow("result", res_plotted)
cv2.waitKey(0)

print(f"Object detection executed in {execution_time:.4f} seconds.")