import io
import os
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import pydicom
from torchvision import transforms

from app.model import MammogramClassifier
from app.gradcam import GradCAM, overlay_heatmap
from app.pacs_client import PACSClient

app = FastAPI(title="Enterprise Mammogram AI API", version="2.2")

model = None
device = None
current_model_name = "resnet50"

PACS_URL = os.getenv("PACS_BASE_URL", "https://pacs.hospital-network.local/dicomweb")
PACS_TOKEN = os.getenv("PACS_AUTH_TOKEN", None)
pacs_client = PACSClient(pacs_base_url=PACS_URL, auth_token=PACS_TOKEN)

pacs_router = APIRouter(prefix="/pacs", tags=["PACS Integration"])

@app.on_event("startup")
def load_default_model():
    global model, device, current_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model_name = os.getenv("MODEL_BACKBONE", "resnet50")
    model = MammogramClassifier(model_name=current_model_name, pretrained=False)
    model.to(device)
    model.eval()

@app.post("/switch_model/")
def switch_model(model_name: str = Query(..., description="Model architecture")):
    global model, device, current_model_name
    if model_name not in ['resnet50', 'efficientnet_b0', 'densenet121', 'vit_b_16']:
        raise HTTPException(status_code=400, detail=f"Unsupported model architecture: {model_name}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MammogramClassifier(model_name=model_name, pretrained=False)
        model.to(device)
        model.eval()
        current_model_name = model_name
        return {"status": "success", "active_model": current_model_name, "device": str(device)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_image(image_bytes: bytes, filename: str, is_vit: bool = False):
    dicom_meta = {}
    target_size = (224, 224) if is_vit else (512, 512)
    
    if filename.endswith(".dcm"):
        dcm = pydicom.dcmread(io.BytesIO(image_bytes))
        img_np = dcm.pixel_array.astype(np.float32)
        dicom_meta['PatientID'] = str(getattr(dcm, 'PatientID', 'N/A'))
        dicom_meta['ViewPosition'] = str(getattr(dcm, 'ViewPosition', 'N/A'))
        dicom_meta['ImageLaterality'] = str(getattr(dcm, 'ImageLaterality', 'N/A'))
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            img_np = np.max(img_np) - img_np
        img_min, img_max = img_np.min(), img_np.max()
        if img_max - img_min > 0:
            img_np = (img_np - img_min) / (img_max - img_min) * 255.0
        img_np = np.uint8(img_np)
        pil_img = Image.fromarray(img_np).convert("RGB")
    else:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        dicom_meta = {'PatientID': 'N/A', 'ViewPosition': 'N/A', 'ImageLaterality': 'N/A'}

    pil_resized = pil_img.resize(target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_resized).unsqueeze(0)
    return tensor, np.array(pil_resized), dicom_meta

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device), "active_model": current_model_name}

@app.post("/predict_with_cam/")
async def predict_with_cam(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        is_vit = (current_model_name == 'vit_b_16')
        tensor, original_np, dicom_meta = preprocess_image(contents, file.filename, is_vit=is_vit)
        input_tensor = tensor.to(device)
        input_tensor.requires_grad = True

        with torch.set_grad_enabled(True):
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            
            if is_vit:
                target_layer = model.backbone.encoder.layers[-1].ln_1
            elif hasattr(model.backbone, 'layer4'):
                target_layer = model.backbone.layer4[-1]
            else:
                target_layer = model.backbone
                
            grad_cam = GradCAM(model, target_layer)
            heatmap = grad_cam.generate_heatmap(input_tensor)

        superimposed = overlay_heatmap(original_np, heatmap)
        result_pil = Image.fromarray(superimposed)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        buf.seek(0)

        prediction_label = "Positive (Malignant/Anomaly)" if prob >= 0.5 else "Negative (Normal/Benign)"
        headers = {
            "X-Malignancy-Probability": str(round(prob, 4)),
            "X-Prediction-Label": prediction_label,
            "X-Active-Model": current_model_name,
            "X-Patient-ID": dicom_meta['PatientID']
        }
        return StreamingResponse(buf, media_type="image/png", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@pacs_router.get("/worklist/")
def get_pacs_worklist(patient_id: str = Query(None)):
    try:
        studies = pacs_client.query_worklist(patient_id=patient_id)
        return {"status": "success", "study_count": len(studies), "studies": studies}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

app.include_router(pacs_router)
