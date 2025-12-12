# Generating predictions from fine-tuned ResNet50 model
import torch
from PIL import Image
from torchvision import transforms as T


# Preprocessing 
preprocess = T.Compose([
    T.Resize((64, 64)),         # Resize to match model input 
    T.ToTensor(),               # Convert to tensor + scale to [0,1]
    T.Normalize(                # Normalization
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict_image(model, image_path, class_names=None):
    """
    Generate predictions from uploaded image 
    """
    # Load image as PIL
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess
    img_tensor = preprocess(img).unsqueeze(0)  # add batch dimension [1,3,64,64]
    
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
