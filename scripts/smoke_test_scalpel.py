import sys
import os
import torch

# Ensure project root is in path
sys.path.append(os.getcwd())

try:
    from core.models.scalpel.arch import LesionScalpel
except ImportError:
    # Fallback if running from scripts dir
    sys.path.append(os.path.dirname(os.getcwd()))
    from core.models.scalpel.arch import LesionScalpel

def smoke_test():
    print("Initializing LesionScalpel...")
    model = LesionScalpel()
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)
    
    # Random tensor
    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
        
    print(f"Output shape: {y.shape}")
    
    # Verification
    assert y.shape == (B, 1, H, W), f"Shape mismatch: {y.shape} != {(B, 1, H, W)}"
    
    print("GPU Smoke Test: [PASSED]")

if __name__ == "__main__":
    smoke_test()
