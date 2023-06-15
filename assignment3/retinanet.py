import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import retinanet_resnet50_fpn

coco_dataset_classes = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "street sign", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
    "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis",
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

# Load pre-trained RetinaNet model
model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Apply transformations to the image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Run object detection on the image
with torch.no_grad():
    predictions = model([image_tensor])

# Display object detection results
threshold = 0.5  # Confidence threshold for showing detection boxes

# Create a drawing object
draw = ImageDraw.Draw(image)

for i in range(len(predictions[0]['boxes'])):
    if predictions[0]['scores'][i] > threshold:
        bbox = predictions[0]['boxes'][i]
        label = predictions[0]['labels'][i]
        confidence = predictions[0]['scores'][i]

        # Get the coordinates of the bounding box
        xmin, ymin, xmax, ymax = bbox

        # Draw the bounding box rectangle
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)

        # Add label and confidence score as text
        label_text = f'{coco_dataset_classes[label.item()]}: {confidence.item():.2f}'
        draw.text((xmin, ymin - 10), label_text, fill='red')

# Show the image with the bounding box annotations
image.show()
