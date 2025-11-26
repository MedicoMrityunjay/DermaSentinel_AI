import os

def configure_server():
    server_code = r'''
import os
import io
import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import timm
import segmentation_models_pytorch as smp
import base64
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

# 1. SETUP & DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Server starting on {DEVICE}...")

# 2. DEFINE MODELS
class DermaFusion(nn.Module):
    def __init__(self, meta_dim=11):
        super().__init__()
        full_model = timm.create_model('tf_efficientnet_b3_ns', pretrained=False, num_classes=1)
        self.eye = full_model
        self.eye.classifier = nn.Identity()
        self.brain = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(1536 + 32, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 1)
        )
    def forward(self, img, meta):
        return self.head(torch.cat((self.eye(img), self.brain(meta)), dim=1))

# 3. INITIALIZE APP
app = FastAPI(title="DermaSentinel 3D", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files (Crucial for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 4. LOAD WEIGHTS
SCALPEL_PATH = r"core/models/scalpel/weights/best_scalpel.pth"
FUSION_PATH = r"core/models/fusion/weights/best_fusion.pth"

scalpel = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
fusion = DermaFusion().to(DEVICE)

def load_weights(model, path):
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=DEVICE)
            if 'model_state_dict' in state: state = state['model_state_dict']
            model.load_state_dict(state)
            model.eval()
            print(f"✅ Loaded {path}")
        except Exception as e: print(f"⚠️ Error loading {path}: {e}")
    else:
        print(f"⚠️ Weights not found: {path}")

load_weights(scalpel, SCALPEL_PATH)
load_weights(fusion, FUSION_PATH)

# 5. TRANSFORMS
transforms = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])

# 6. ENDPOINTS

@app.get("/")
def home():
    # Serve the 3D Frontend
    return FileResponse('static/index.html')

@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    site: str = Form(...)
):
    # READ
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    # PREPROCESS
    augmented = transforms(image=image_np)["image"].unsqueeze(0).to(DEVICE)
    
    # GATE 1: SCALPEL
    with torch.no_grad():
        mask_prob = torch.sigmoid(scalpel(augmented))
        mask = (mask_prob > 0.5).float()
        
    coverage = mask.sum().item() / (512*512)
    
    # GENERATE OVERLAY (Base64)
    mask_np = mask[0, 0].cpu().numpy() # (512, 512)
    
    # Create Cyan Overlay for 3D Theme
    overlay_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    overlay_rgba[:, :, 1] = 255 # Green/Cyan mix
    overlay_rgba[:, :, 2] = 255 # Blue
    overlay_rgba[:, :, 3] = (mask_np * 100).astype(np.uint8) # Alpha
    
    overlay_img = Image.fromarray(overlay_rgba)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    if coverage < 0.01:
        return {
            "status": "REJECTED",
            "reason": "No lesion detected",
            "mask_coverage": coverage,
            "mask_base64": mask_b64
        }

    # GATE 2: FUSION
    meta_tensor = torch.zeros(1, 11).to(DEVICE)
    meta_tensor[0, 0] = (age - 50) / 20.0
    
    with torch.no_grad():
        prob = torch.sigmoid(fusion(augmented, meta_tensor)).item()
        
    diagnosis = "Melanoma" if prob > 0.5 else "Benign"
    severity = "High" if prob > 0.8 else "Low"
    if 0.5 < prob <= 0.8: severity = "Medium"

    return {
        "status": "SUCCESS",
        "diagnosis": diagnosis,
        "probability": prob,
        "severity": severity,
        "segmentation_coverage": coverage,
        "mask_base64": mask_b64
    }
'''
    with open("main_server.py", "w", encoding="utf-8") as f:
        f.write(server_code)
    print("Backend Configured for 3D Frontend.")

if __name__ == "__main__":
    configure_server()
