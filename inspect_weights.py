import torch
import os

def inspect():
    path = "core/models/fusion/weights/best_fusion.pth"
    if not os.path.exists(path):
        print("No weights found.")
        return

    try:
        state_dict = torch.load(path, map_location='cpu')
        print("Keys found:")
        for k, v in list(state_dict.items())[:20]: # Print first 20
            print(f"{k}: {v.shape}")
            
        # Check for specific layers to infer arch
        if "backbone.conv_stem.weight" in state_dict:
            print("Looks like EfficientNet or similar.")
        if "meta_net.0.weight" in state_dict:
            print("Has metadata net.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
