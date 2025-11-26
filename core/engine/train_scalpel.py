import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.models.scalpel.arch import LesionScalpel
from core.data.seg_dataset import SegmentationDataset, get_train_transforms, get_val_transforms


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


def calculate_dice_score(pred, target, threshold=0.5):
    """Calculate Dice Score"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            dice = calculate_dice_score(outputs, masks)
            iou = calculate_iou(outputs, masks)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
        
        pbar.set_postfix({'loss': loss.item(), 'dice': dice.item(), 'iou': iou.item()})
    
    return running_loss / len(dataloader), np.mean(dice_scores), np.mean(iou_scores)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    
    pbar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            dice = calculate_dice_score(outputs, masks)
            iou = calculate_iou(outputs, masks)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice.item(), 'iou': iou.item()})
    
    return running_loss / len(dataloader), np.mean(dice_scores), np.mean(iou_scores)


def train_scalpel(train_dataset, val_dataset, epochs=10, batch_size=4, lr=1e-4, device='cuda'):
    """
    Main training function for the Scalpel model
    """
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = LesionScalpel().to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Create weights directory
    weights_dir = "core/models/scalpel/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {device}...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(weights_dir, "best_scalpel.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    return model


if __name__ == "__main__":
    # Example usage with mock data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = SegmentationDataset([], [], transforms=get_train_transforms(), mock_mode=True)
    val_dataset = SegmentationDataset([], [], transforms=get_val_transforms(), mock_mode=True)
    
    model = train_scalpel(train_dataset, val_dataset, epochs=2, batch_size=4, device=device)
    print("Training complete!")
