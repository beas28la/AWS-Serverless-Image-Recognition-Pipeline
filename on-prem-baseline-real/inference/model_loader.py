# Script to laod in the fine-tuned RESNET model 

# Steps
# 1. Define the model architecture 
# 2. Load the state_dict into the model 
# 3. Call model.eval() and return model 

import torch 
import torch.nn as nn
import torchvision.models as tv_models

def load_resnet_model(MODEL_PATH, num_classes=10, device='cpu'):
    """
    Load fine-tuned ResNet50 from state_dict saved in .pth file
    """
    # ------------------------
    # Define the architecture 
    # ------------------------
    model = tv_models.resnet50(weights=None)
    # Replace the final FC layer
    in_features = model.fc.in_features
    
    # Replace final layer with correct number of classes for Eurosat (i.e., 10)
    model.fc = nn.Linear(in_features, num_classes)

    # ----------------------------
    # Load the saved weights 
    # ------------------------------- 
    try: 
        state_dict = torch.load(MODEL_PATH, map_location=device)
        # Set strict=False to ignore any missing or extra keys 
        # 
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Fine-tuned ResNet50 loaded successfully.")
        return model
    except Exception as e: 
        print(f"Failed to load fine-tuned ResNet50 model: {e}")
        return None
    