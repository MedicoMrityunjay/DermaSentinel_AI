import os
import logging
from huggingface_hub import hf_hub_download
REPO_ID = "medicomrityunjay/DermaSentinel-Weights"

import io
import cv2
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import timm
import segmentation_models_pytorch as smp
import base64
import uuid
import json
import shutil
import time
import math
from datetime import datetime, timedelta
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, Form, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from pydantic import BaseModel
from typing import Optional, List
from transformers import BlipProcessor, BlipForConditionalGeneration
from sqlalchemy.orm import Session
from core.database import SessionLocal, engine, get_db
from core.models.db import Base, Scan, Patient

# 1. SETUP & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Server starting on {DEVICE}...")

# Ensure dirs exist
os.makedirs("static/temp", exist_ok=True)
os.makedirs("data/active_learning/images", exist_ok=True)

# Initialize DB
Base.metadata.create_all(bind=engine)

# 3. DEFINE MODELS
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

# 4. INITIALIZE APP
app = FastAPI(title="DermaSentinel Research", version="13.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://medicomrityunjay-dermasentinel.hf.space",
        "http://localhost:7860",
        "http://127.0.0.1:7860"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# CLEANUP ON STARTUP
@app.on_event("startup")
async def startup_event():
    temp_dir = "static/temp"
    if os.path.exists(temp_dir):
        count = 0
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        logger.info(f"🧹 Cleaned {count} files from temp directory.")

# 5. LOAD WEIGHTS
logger.info("⏳ Downloading Scalpel from HF...")
SCALPEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="best_scalpel.pth")
logger.info("⏳ Downloading Fusion from HF...")
FUSION_PATH = hf_hub_download(repo_id=REPO_ID, filename="best_fusion.pth")

# HELPER FUNCTION FOR LOADING
def load_weights_smart(model, path, device):
    try:
        state_dict = torch.load(path, map_location=device, weights_only=False)
        # Handle case where state_dict is nested under 'model_state_dict'
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Fix "model." prefix issue
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k.replace("model.", "", 1)] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info(f"✅ Loaded {path}")
    except Exception as e:
        logger.error(f"⚠️ Error loading {path}: {e}")

logger.info("⏳ Loading Models...")
scalpel = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
load_weights_smart(scalpel, SCALPEL_PATH, DEVICE)

fusion = DermaFusion().to(DEVICE)
load_weights_smart(fusion, FUSION_PATH, DEVICE)

# Load BLIP (GPU ACCELERATED)
logger.info("⏳ Loading AI Scribe (BLIP)...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    logger.info(f"✅ AI Scribe Ready on {DEVICE}")
except Exception as e:
    logger.error(f"⚠️ BLIP Load Error: {e}")
    blip_processor = None
    blip_model = None

# 6. TRANSFORMS & TTA
base_transforms = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])

def get_tta_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

# 7. HELPERS
def analyze_concepts(image_np, mask_np):
    mask_u8 = (mask_np * 255).astype(np.uint8)
    
    # Asymmetry
    rotated = cv2.rotate(mask_u8, cv2.ROTATE_180)
    intersection = np.logical_and(mask_u8 > 127, rotated > 127).sum()
    union = np.logical_or(mask_u8 > 127, rotated > 127).sum()
    iou = intersection / (union + 1e-6)
    asymmetry_score = 1.0 - iou
    
    # Border
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area > 0:
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            border_score = 1.0 - (1.0 / max(1.0, compactness))
        else:
            border_score = 0.0
    else:
        border_score = 0.0
        
    # Color
    mask_bool = mask_u8 > 127
    if mask_bool.sum() > 0:
        pixels = image_np[mask_bool]
        std_r = np.std(pixels[:, 0])
        std_g = np.std(pixels[:, 1])
        std_b = np.std(pixels[:, 2])
        avg_std = (std_r + std_g + std_b) / 3.0
        color_score = min(1.0, avg_std / 60.0)
    else:
        color_score = 0.0
        
    return {
        "asymmetry": float(asymmetry_score),
        "border": float(border_score),
        "color": float(color_score)
    }

def analyze_skin_tone(image_np, mask_np):
    """
    Analyzes skin tone using Individual Typology Angle (ITA) in Lab color space.
    ITA = arctan((L - 50) / b) * (180 / pi)
    """
    # Convert to Lab
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2Lab)
    
    # Vectorized Mean with Mask
    bg_mask = ((mask_np < 0.5) * 255).astype(np.uint8)
    
    if cv2.countNonZero(bg_mask) == 0:
        return "Unknown", 0.0, False
        
    mean_vals = cv2.mean(lab_image, mask=bg_mask)
    L_cv = mean_vals[0]
    b_cv = mean_vals[2]
    
    L_std = L_cv * (100.0 / 255.0)
    b_std = b_cv - 128.0
    
    # Calculate ITA
    try:
        ita = math.atan((L_std - 50.0) / (b_std + 1e-6)) * (180.0 / math.pi)
    except:
        ita = 0.0
        
    # Classify
    if ita > 55:
        skin_type = "Very Light (Type I)"
    elif 41 < ita <= 55:
        skin_type = "Light (Type II)"
    elif 28 < ita <= 41:
        skin_type = "Intermediate (Type III)"
    elif 10 < ita <= 28:
        skin_type = "Tan (Type IV)"
    elif -30 < ita <= 10:
        skin_type = "Brown (Type V)"
    else:
        skin_type = "Dark (Type VI)"
        
    # Bias Warning for Type V/VI
    bias_warning = ita <= 10
    
    return skin_type, float(ita), bias_warning

def generate_clinical_note(image, abcd, diagnosis, prob, skin_type):
    note = ""
    # BLIP Caption
    if blip_processor and blip_model:
        try:
            inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            note += f"Visual analysis reveals {caption}. "
        except Exception as e:
            logger.error(f"BLIP Error: {e}")
            pass
    
    # ABCD Synthesis
    if abcd['asymmetry'] > 0.5:
        note += "The lesion exhibits significant asymmetry. "
    else:
        note += "The lesion is relatively symmetric. "
        
    if abcd['border'] > 0.5:
        note += "Borders appear irregular and ill-defined. "
    else:
        note += "Borders are well-circumscribed. "
        
    if abcd['color'] > 0.5:
        note += "There is notable color variegation. "
    
    # Skin Tone Context
    note += f"Patient skin phenotype is assessed as {skin_type}. "
    
    # Diagnosis Context
    note += f"AI diagnostic assessment suggests {diagnosis} (Confidence: {prob*100:.1f}%). "
    
    if diagnosis == "Melanoma":
        note += "Immediate clinical correlation is recommended."
    else:
        note += "Routine monitoring is advised."
        
    return note

# 9. ENDPOINTS

@app.get("/")
def home():
    return FileResponse('static/index.html')

@app.post("/diagnose")
async def diagnose(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    site: str = Form(...),
    db: Session = Depends(get_db)
):
    # VALIDATION
    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age. Must be between 0 and 120.")
    
    # READ
    contents = await file.read()
    
    # SIZE CHECK (10MB Limit)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    
    # QUALITY GATE: BLUR DETECTION
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if variance < 100:
        return {
            "status": "REJECTED",
            "reason": f"Image too blurry (Score: {int(variance)} < 100). Please retake.",
            "blur_score": variance
        }

    # GATE 1: SCALPEL
    augmented_base = base_transforms(image=image_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_prob = torch.sigmoid(scalpel(augmented_base))
        mask = (mask_prob > 0.5).float()
        
    coverage = mask.sum().item() / (512*512)
    
    # GENERATE OVERLAY
    mask_np = mask[0, 0].cpu().numpy()
    overlay_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    overlay_rgba[:, :, 1] = 255 # Cyan
    overlay_rgba[:, :, 2] = 255
    overlay_rgba[:, :, 3] = (mask_np * 100).astype(np.uint8)
    
    overlay_img = Image.fromarray(overlay_rgba)
    
    # Save for Report
    img_resized = image.resize((512, 512))
    img_resized.paste(overlay_img, (0, 0), overlay_img)
    
    scan_id = str(uuid.uuid4())
    save_path = f"static/temp/{scan_id}.png"
    img_resized.save(save_path)
    
    # Base64
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

    # CONCEPT ANALYSIS (ABCD)
    img_for_analysis = cv2.resize(image_np, (512, 512))
    concepts = analyze_concepts(img_for_analysis, mask_np)
    
    # EQUITY ENGINE (Skin Tone)
    skin_type, ita_score, bias_warning = analyze_skin_tone(img_for_analysis, mask_np)

    # GATE 2: FUSION + TTA (OPTIMIZED)
    meta_tensor = torch.zeros(1, 11).to(DEVICE)
    meta_tensor[0, 0] = (age - 50) / 20.0
    
    tta_aug = get_tta_transforms()
    
    # Batch: 1 Original + 4 Augmented
    batch_images = [augmented_base]
    for _ in range(4):
        aug_img = tta_aug(image=image_np)["image"].unsqueeze(0).to(DEVICE)
        batch_images.append(aug_img)
        
    batch_tensor = torch.cat(batch_images, dim=0)
    meta_batch = meta_tensor.repeat(5, 1)
    
    with torch.no_grad():
        batch_probs = torch.sigmoid(fusion(batch_tensor, meta_batch))
        predictions = batch_probs.cpu().numpy().flatten().tolist()
            
    final_prob = np.mean(predictions)
    uncertainty = np.std(predictions)
    
    diagnosis = "Melanoma" if final_prob > 0.5 else "Benign"
    severity = "High" if final_prob > 0.8 else "Low"
    if 0.5 < final_prob <= 0.8: severity = "Medium"

    # AI SCRIBE
    clinical_note = generate_clinical_note(image, concepts, diagnosis, final_prob, skin_type)

    # SAVE TO DB
    metadata = {
        "concepts": concepts,
        "skin_type": skin_type,
        "ita_score": ita_score,
        "bias_warning": bias_warning,
        "clinical_note": clinical_note,
        "tta_predictions": predictions
    }
    
    new_scan = Scan(
        id=scan_id,
        diagnosis=diagnosis,
        probability=final_prob,
        severity=severity,
        uncertainty=uncertainty,
        age=age,
        sex=sex,
        site=site,
        metadata_json=json.dumps(metadata)
    )
    db.add(new_scan)
    db.commit()

    return {
        "status": "SUCCESS",
        "scan_id": scan_id,
        "diagnosis": diagnosis,
        "probability": final_prob,
        "uncertainty_score": uncertainty,
        "severity": severity,
        "segmentation_coverage": coverage,
        "mask_base64": mask_b64,
        "concepts": concepts,
        "skin_type": skin_type,
        "ita_score": ita_score,
        "bias_warning": bias_warning,
        "clinical_note": clinical_note,
        "tta_predictions": predictions
    }

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    # Return last 10 scans (Global History)
    scans = db.query(Scan).order_by(Scan.timestamp.desc()).limit(10).all()
    return [
        {
            "id": s.id,
            "diagnosis": s.diagnosis,
            "date": s.timestamp.strftime("%Y-%m-%d %H:%M"),
            "severity": s.severity
        } 
        for s in scans
    ]

@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # READ
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    if not blip_processor or not blip_model:
        return {"answer": "AI Consultant is offline (Model not loaded)."}
        
    try:
        # VQA Prompt
        text = f"Question: {question} Answer:"
        inputs = blip_processor(image, text, return_tensors="pt").to(DEVICE)
        
        out = blip_model.generate(**inputs)
        answer = blip_processor.decode(out[0], skip_special_tokens=True)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

@app.get("/report/{scan_id}")
def generate_report(scan_id: str, db: Session = Depends(get_db)):
    img_path = f"static/temp/{scan_id}.png"
    if not os.path.exists(img_path):
        return JSONResponse(status_code=404, content={"message": "Scan image not found"})
        
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        return JSONResponse(status_code=404, content={"message": "Scan record not found"})
    
    meta = json.loads(scan.metadata_json)
    
    pdf_path = f"static/temp/{scan_id}.pdf"
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFillColor(colors.navy)
    c.rect(0, height - 80, width, 80, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, height - 50, "DERMASENTINEL DIAGNOSTICS")
    c.setFont("Helvetica", 10)
    c.drawString(40, height - 70, f"Clinical AI Report | ID: {scan_id} | {scan.timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    # Layout
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.gray)
    c.line(40, height - 100, width - 40, height - 100)
    
    # Metadata
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 130, "Patient Metadata")
    c.setFont("Helvetica", 12)
    y_meta = height - 160
    c.drawString(40, y_meta, f"Age: {scan.age}")
    c.drawString(40, y_meta - 20, f"Sex: {scan.sex.title()}")
    c.drawString(40, y_meta - 40, f"Anatomic Site: {scan.site.title()}")
    c.drawString(40, y_meta - 60, f"Skin Phenotype: {meta.get('skin_type', 'N/A')}")
    if meta.get('bias_warning'):
        c.setFillColor(colors.red)
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(40, y_meta - 75, "Warning: High melanin content detected.")
        c.drawString(40, y_meta - 85, "Segmentation accuracy may be affected.")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
    
    # Findings
    c.setFont("Helvetica-Bold", 14)
    c.drawString(300, height - 130, "AI Analysis Findings")
    diagnosis = scan.diagnosis
    prob = scan.probability
    uncertainty = scan.uncertainty
    concepts = meta.get('concepts', {})
    
    c.setFont("Helvetica-Bold", 18)
    if diagnosis == "Melanoma":
        c.setFillColor(colors.red)
    else:
        c.setFillColor(colors.green)
    c.drawString(300, y_meta, f"{diagnosis.upper()}")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(300, y_meta - 25, f"Malignancy Probability: {prob*100:.1f}%")
    c.drawString(300, y_meta - 45, f"Confidence Interval: ±{uncertainty*100:.1f}%")
    
    # ABCD
    c.setFont("Helvetica-Bold", 12)
    c.drawString(300, y_meta - 80, "Clinical Concepts (ABCD)")
    c.setFont("Helvetica", 10)
    c.drawString(300, y_meta - 100, f"Asymmetry: {concepts.get('asymmetry', 0)*100:.1f}%")
    c.drawString(300, y_meta - 115, f"Border Irregularity: {concepts.get('border', 0)*100:.1f}%")
    c.drawString(300, y_meta - 130, f"Color Variation: {concepts.get('color', 0)*100:.1f}%")
    
    # Bar
    bar_x = 300
    bar_y = y_meta - 160
    bar_w = 200
    bar_h = 15
    c.setFillColor(colors.lightgrey)
    c.rect(bar_x, bar_y, bar_w, bar_h, fill=1, stroke=0)
    fill_w = bar_w * prob
    if prob > 0.5:
        c.setFillColor(colors.red)
    else:
        c.setFillColor(colors.green)
    c.rect(bar_x, bar_y, fill_w, bar_h, fill=1, stroke=0)
    c.setStrokeColor(colors.black)
    c.rect(bar_x, bar_y, bar_w, bar_h, fill=0, stroke=1)
    
    # Image
    c.drawImage(img_path, 40, height - 550, width=250, height=250, preserveAspectRatio=True)
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.gray)
    c.drawString(40, height - 565, "Visualized Lesion with Segmentation Overlay")
    
    # Clinical Note
    note = meta.get('clinical_note', '')
    if note:
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, height - 600, "AI Clinical Note")
        c.setFont("Helvetica", 10)
        # Simple text wrap logic for PDF
        text_obj = c.beginText(40, height - 620)
        words = note.split()
        line = ""
        for word in words:
            if c.stringWidth(line + word, "Helvetica", 10) < 500:
                line += word + " "
            else:
                text_obj.textLine(line)
                line = word + " "
        text_obj.textLine(line)
        c.drawText(text_obj)
    
    # Footer
    c.setStrokeColor(colors.lightgrey)
    c.line(40, 50, width - 40, 50)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.darkgrey)
    c.drawString(40, 35, "Disclaimer: This report is generated by an Artificial Intelligence system (DermaSentinel v13.0).")
    c.drawString(40, 25, "It is intended for clinical decision support only and does not constitute a definitive medical diagnosis.")
    c.drawString(width - 200, 35, "Powered by DermaSentinel")
    
    c.save()
    
    return FileResponse(pdf_path, media_type='application/pdf', filename=f"DermaSentinel_Report_{scan_id}.pdf")

class FeedbackModel(BaseModel):
    scan_id: str
    agreement: bool
    correct_diagnosis: Optional[str] = None
    notes: Optional[str] = None

@app.post("/feedback")
def submit_feedback(feedback: FeedbackModel):
    # Feedback logic remains file-based for now or can be migrated later
    scan_id = feedback.scan_id
    temp_img_path = f"static/temp/{scan_id}.png"
    if not os.path.exists(temp_img_path):
        return JSONResponse(status_code=404, content={"message": "Scan data not found"})
        
    # In a real app, we'd update the DB record here too
    
    record = {
        "scan_id": scan_id,
        "doctor_agreement": feedback.agreement,
        "doctor_diagnosis": feedback.correct_diagnosis,
        "doctor_notes": feedback.notes,
        "feedback_timestamp": datetime.now().isoformat()
    }
    
    timestamp = int(time.time())
    filename = f"{timestamp}_{scan_id}"
    
    with open(f"data/active_learning/{filename}.json", "w") as f:
        json.dump(record, f, indent=4)
        
    shutil.copy(temp_img_path, f"data/active_learning/images/{filename}.png")
    
    return {"status": "Saved", "message": "Feedback recorded for future training."}
