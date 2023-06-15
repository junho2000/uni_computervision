import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fcos_resnet50_fpn, retinanet_resnet50_fpn, ssdlite320_mobilenet_v3_large
import time

# Define the list of class labels
cocodataset_class_labels = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# loads pre-trained object detection model using the TorchVision module of PyTorch
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
fcos_model = fcos_resnet50_fpn(pretrained=True)
retinanet_model = retinanet_resnet50_fpn(pretrained=True)
ssd_model = ssdlite320_mobilenet_v3_large(pretrained=True)

frcnn_model.eval()
fcos_model.eval()
retinanet_model.eval()
ssd_model.eval()

#font = ImageFont.truetype("/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A3/Roboto-Black.ttf", 25) #폰트를 다운로드 받아야 글자크기를 조절할 수 있음

def visualize(draw, predictions, threshold, model_name):
    for i in range(len(predictions[0]['scores'])):
        if predictions[0]['scores'][i] > threshold:
            bbox = predictions[0]['boxes'][i]
            label = predictions[0]['labels'][i]
            confidence = predictions[0]['scores'][i]
            
            print(f"{model_name} {cocodataset_class_labels[label]} {confidence:.2f}")
            
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])] ,outline="red", width = 2)
            #draw.text((bbox[0], bbox[1]), f"{model_name} {cocodataset_class_labels[label]} {confidence:.2f}", fill="red", font=font) #다운받은 폰트를 사용
            draw.text((bbox[0], bbox[1]), f"{model_name} {cocodataset_class_labels[label]} {confidence:.2f}", fill="red")

# loads an image using PIL
image_path = "/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A3/traffic-light-car-copy.jpg"
frcnn_image = Image.open(image_path)
fcos_image = Image.open(image_path)
retinanet_image = Image.open(image_path)
ssd_image = Image.open(image_path)

transform = transforms.Compose([transforms.ToTensor()])
frcnn_image_tensor = transform(frcnn_image)
fcos_image_tensor = transform(frcnn_image)
retinanet_image_tensor = transform(frcnn_image)
ssd_image_tensor = transform(frcnn_image)

# runs object detection using the pre-trained model on the loaded image and measure time duration
#faster rcnn

start_time = time.time()
with torch.no_grad():
    frcnn_predictions = frcnn_model([frcnn_image_tensor])
end_time = time.time()
frcnn_execution_time = end_time - start_time
print(f"Object detection executed in {frcnn_execution_time:.4f} seconds with frcnn")

#fcos
start_time = time.time()
with torch.no_grad():
    fcos_predictions = fcos_model([fcos_image_tensor])
end_time = time.time()
fcos_execution_time = end_time - start_time
print(f"Object detection executed in {fcos_execution_time:.4f} seconds with fcos")

#retina net
start_time = time.time()
with torch.no_grad():
    retinanet_predictions = retinanet_model([retinanet_image_tensor])
end_time = time.time()
retinanet_execution_time = end_time - start_time
print(f"Object detection executed in {retinanet_execution_time:.4f} seconds with retinanet")

#ssd
start_time = time.time()
with torch.no_grad():
    ssd_predictions = ssd_model([ssd_image_tensor])
end_time = time.time()
ssd_execution_time = end_time - start_time
print(f"Object detection executed in {ssd_execution_time:.4f} seconds with ssd")

# displays the object detection results (bounding box, object label, confidence value) overlayed on the loaded image
threshold = 0.8  # 신뢰도 

frcnn_draw = ImageDraw.Draw(frcnn_image)
fcos_draw = ImageDraw.Draw(fcos_image)
retinanet_draw = ImageDraw.Draw(retinanet_image)
ssd_draw = ImageDraw.Draw(ssd_image)

visualize(frcnn_draw, frcnn_predictions, threshold, "frcnn") #그림을 그릴 draw 객체, 모델별 추론 결과, 임계값, 모델명
visualize(fcos_draw, fcos_predictions, threshold, "fcos")
visualize(retinanet_draw, retinanet_predictions, threshold, "retinanet")
visualize(ssd_draw, ssd_predictions, threshold, "ssd")

frcnn_image.show()
fcos_image.show()
retinanet_image.show()
ssd_image.show()