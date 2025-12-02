# Generating predictions from fine-tuned ResNet50 model
import torch
from PIL import Image
from torchvision.io import read_image

from torchvision import transforms as T

# Preprocessing (normalize images)
# Note: Training used original 64x64 EuroSAT images without resize
# So inference should also use 64x64 to match training
preprocess = T.Compose([
                T.Resize((64, 64)),  # Match training image size
                T.ToTensor(),        # Convert to [0, 1] range
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

def predict_image(model, image_path, class_names=None):
    """
    Generate predictions from uploaded image 
    Inputs: 
    - model: fine-tuned ResNet50 model 
    - image_path: full filepath to image 

    Outputs: 
     - confidence: confidence of inference 
     - predicted_idx: predicted class label 

    """
    # Load image as PIL
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess
    img_tensor = preprocess(img).unsqueeze(0)  # 4D: [1,3,H,W]
    
    # Forward pass for inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_idx = predicted_idx.item()
    confidence = float(confidence.item())

    if class_names is not None:
        return class_names[predicted_idx], confidence
    else:
        return predicted_idx, confidence
