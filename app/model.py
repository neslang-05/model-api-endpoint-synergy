import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

CLASSES = ['impure', 'pure']


def _clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        if new_key.startswith('model.'):
            new_key = new_key[len('model.'):]
        cleaned[new_key] = value
    return cleaned


def _extract_checkpoint_details(checkpoint):
    architecture = None
    class_names = None
    num_classes = None

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        architecture = checkpoint.get('architecture')
        class_names = checkpoint.get('class_names')
        num_classes = checkpoint.get('num_classes')
    else:
        state_dict = checkpoint

    state_dict = _clean_state_dict_keys(state_dict)
    return state_dict, architecture, class_names, num_classes


def _infer_num_classes(state_dict, class_names, num_classes):
    if isinstance(num_classes, int) and num_classes > 0:
        return num_classes

    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        return len(class_names)

    for key in ('fc.4.weight', 'fc.weight'):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, 'shape') and len(tensor.shape) >= 1:
            return int(tensor.shape[0])

    return len(CLASSES)


def _build_model(architecture, num_classes):
    architecture_name = str(architecture or 'resnet50').lower()
    supported_architectures = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    model_factory = supported_architectures.get(architecture_name, models.resnet50)
    model = model_factory(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def get_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict, architecture, class_names, num_classes = _extract_checkpoint_details(checkpoint)
    resolved_num_classes = _infer_num_classes(state_dict, class_names, num_classes)

    model = _build_model(architecture, resolved_num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if isinstance(class_names, (list, tuple)) and len(class_names) == resolved_num_classes:
        model.class_names = [str(name) for name in class_names]
    elif len(CLASSES) == resolved_num_classes:
        model.class_names = CLASSES
    else:
        model.class_names = [f'class_{index}' for index in range(resolved_num_classes)]

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

        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence = probabilities[predicted_idx].item()

    class_names = getattr(model, 'class_names', CLASSES)
    if predicted_idx < len(class_names):
        prediction = class_names[predicted_idx]
    else:
        prediction = str(predicted_idx)

    return prediction, confidence
