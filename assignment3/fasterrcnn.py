import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import time

# Define the list of class labels
fasterrcnn_class_labels = [
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

# Load image using PIL
image_path = "/Users/kimjunho/Desktop/컴퓨터비전3-1/[CV]A3/img_example.JPG"
image = Image.open(image_path)

# Load pre-trained object detection model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Apply transformations to the image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Run object detection and measure time duration
start_time = time.time()
with torch.no_grad():
    predictions = model([image_tensor])
end_time = time.time()
execution_time = end_time - start_time

# Display object detection results
threshold = 0.5  # Confidence threshold for showing detection boxes
draw = ImageDraw.Draw(image)
for i in range(len(predictions[0]['scores'])):
    if predictions[0]['scores'][i] > threshold:
        bbox = predictions[0]['boxes'][i]
        label = predictions[0]['labels'][i]
        confidence = predictions[0]['scores'][i]
        
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")
        draw.text((bbox[0], bbox[1]), f"{fasterrcnn_class_labels[label]} {confidence}", fill="red")

image.show()

print(f"Object detection executed in {execution_time:.4f} seconds.")