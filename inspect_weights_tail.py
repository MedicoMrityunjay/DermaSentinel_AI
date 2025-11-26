import torch
import os

def inspect():
    path = "core/models/fusion/weights/best_fusion.pth"
    try:
        state_dict = torch.load(path, map_location='cpu')
        print("Tail Keys found:")
        keys = list(state_dict.keys())
        for k in keys[-20:]: # Print last 20
            print(f"{k}: {state_dict[k].shape}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
