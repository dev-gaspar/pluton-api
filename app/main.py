# /app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms
from .model_loader import load_model, predict

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = load_model()

# Preprocesamiento de la imagen igual que en entrenamiento
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        lambda img: img.convert("RGB") if img.mode != "RGB" else img,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)
        prediction = predict(model, tensor)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
