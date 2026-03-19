import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Assumed classes based on the description ('pure' and 'impure')
CLASSES = ['impure', 'pure']

def get_model(model_path):
    # Initialize ResNet50 (Not pretrained, as we are loading tuned weights)
    # The torchvision API changed slightly, but models.resnet50() works. 
    # Using older style for compatibility.
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    
    # Custom Head matching: Dropout -> Linear(256) -> ReLU -> Dropout -> Linear(NUM_CLASSES)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def transform_image(image_bytes):
    # Transforms matching the ImageNet statistics and 224x224 resize
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

def get_prediction(model, image_bytes):
    tensor = transform_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = y_hat.item()
        
        # Calculate confidences just in case you need them later
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence = probabilities[predicted_idx].item()
        
    return CLASSES[predicted_idx], confidence
