import torch
from torchvision import transforms
from PIL import Image

from models.models import convert, Darknet

path_to_image = "/home/c3po/Documents/project/learning/amar-works/PyTorch_YOLOv4_meamarp/data/samples/zidane.jpg"
path_to_yolo_cfg = "/home/c3po/Documents/project/learning/amar-works/PyTorch_YOLOv4_meamarp/cfg/yolov3-spp.cfg"
path_converted_pytorch = "/home/c3po/Documents/project/learning/amar-works/PyTorch_YOLOv4_meamarp/yolov3_converted.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(path_to_yolo_cfg)
model.load_state_dict(torch.load(path_converted_pytorch))
model.eval()

# Define the transformations. Adjust as necessary.
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),  # This also scales pixels between [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load an image
image = Image.open(path_to_image)
image = transform(image).unsqueeze(0)  # Add batch dimension
with torch.no_grad():  # No need to track gradients
    predictions = model(image)
print(predictions)

