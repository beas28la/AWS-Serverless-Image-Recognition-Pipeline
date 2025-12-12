# Script to load in the fine-tuned RESNET model 

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
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        
        # Check if state_dict is in PyTorch Lightning format (keys have "model." prefix)
        # If so, remove the prefix to match plain ResNet50 architecture
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("model."):
            print("Detected PyTorch Lightning format, removing 'model.' prefix...")
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() 
                         if k.startswith("model.")}
        
        # Check if state_dict keys match model architecture
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {len(unexpected_keys)} keys")
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"Fine-tuned ResNet50 loaded successfully from {MODEL_PATH}")
        print(f"Loaded {len(loaded_keys & model_keys)} / {len(model_keys)} parameters")
        return model
    except Exception as e: 
        print(f"Failed to load fine-tuned ResNet50 model: {e}")
        return None
    