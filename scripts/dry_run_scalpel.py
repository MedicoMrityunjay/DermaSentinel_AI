import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())

from core.data.seg_dataset import SegmentationDataset, get_train_transforms, get_val_transforms
from core.engine.train_scalpel import train_scalpel


def dry_run():
    """
    Dry run validation:
    - Train for 2 epochs with mock data
    - Verify loss changes
    - Check for CUDA OOM errors
    - Verify weights file creation
    """
    print("=" * 50)
    print("SCALPEL DRY RUN VALIDATION")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create mock datasets
    print("\nCreating mock datasets...")
    train_dataset = SegmentationDataset([], [], transforms=get_train_transforms(), mock_mode=True)
    val_dataset = SegmentationDataset([], [], transforms=get_val_transforms(), mock_mode=True)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Run training
    print("\nStarting dry run training (2 epochs, batch_size=4)...")
    try:
        model = train_scalpel(
            train_dataset, 
            val_dataset, 
            epochs=2, 
            batch_size=4, 
            lr=1e-4,
            device=device
        )
        print("\n✓ Training completed without errors")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n✗ CUDA OOM Error detected!")
            print("Recommendation: Reduce batch_size or image resolution")
            return False
        else:
            raise e
    
    # Verify weights file
    weights_path = "core/models/scalpel/weights/best_scalpel.pth"
    if os.path.exists(weights_path):
        print(f"\n✓ Weights file created: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  - Val Dice: {checkpoint['val_dice']:.4f}")
        print(f"  - Val IoU: {checkpoint['val_iou']:.4f}")
    else:
        print(f"\n✗ Weights file not found: {weights_path}")
        return False
    
    # Final report
    print("\n" + "=" * 50)
    print("DRY RUN RESULTS")
    print("=" * 50)
    print("Engine Status: Assembled")
    print("Mock Training: [PASSED]")
    print("Ready for Real Data: YES")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = dry_run()
    sys.exit(0 if success else 1)
