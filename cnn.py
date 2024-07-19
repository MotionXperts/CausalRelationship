import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet50(pretrained=True)
model.eval()

image = Image.open('/home/peihsin/thesis/dataset/footboard/images/train/000001_jpg.rf.0925f9e3e74d80c6d9af72fa41616eb9.jpg')

transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  

features = torch.nn.Sequential(*list(model.children())[:-1])(input_batch)
features = features.flatten(start_dim=1)  

projection = torch.nn.Linear(2048, 768)
embedded_features = projection(features)
print(embedded_features.shape)
