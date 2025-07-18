# /app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms
from .model_loader import load_model, predict
import sqlite3
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import cloudinary
import cloudinary.uploader
import uuid
import dotenv
from dotenv import load_dotenv

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

# Cargar variables de entorno desde .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_PATH = os.getenv(
    "DB_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "analisis.db")
)
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Configuración de Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
)


# Inicializar la base de datos si no existe
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS analisis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT,
        filename TEXT,
        clase TEXT,
        probabilidad REAL,
        device_id TEXT,
        image_url TEXT
    )"""
    )
    conn.commit()
    conn.close()


init_db()


# Guardar análisis en la base de datos
def guardar_analisis(filename, clase, probabilidad, device_id, image_url):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO analisis (fecha, filename, clase, probabilidad, device_id, image_url) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(),
            filename,
            clase,
            probabilidad,
            device_id,
            image_url,
        ),
    )
    conn.commit()
    conn.close()


@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    device_id: str = Query(..., description="ID del dispositivo"),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0)
        prediction = predict(model, tensor)
        # Calcular probabilidad máxima
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob = float(probabilities.max().item())
        # Subir imagen a Cloudinary
        file.file.seek(0)
        upload_result = cloudinary.uploader.upload(
            file.file,
            folder="pluton",
            public_id=None,
            overwrite=True,
            resource_type="image",
        )
        image_url = upload_result.get("secure_url")
        # Guardar en la base de datos
        guardar_analisis(file.filename, prediction, prob, device_id, image_url)
        return {"prediction": prediction, "probabilidad": prob, "image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint para generar PDF de reportes en un rango de fechas
@app.get("/reporte/pdf")
def generar_reporte_pdf(
    fecha_inicio: str = Query(..., description="Fecha inicio en formato YYYY-MM-DD"),
    fecha_fin: str = Query(..., description="Fecha fin en formato YYYY-MM-DD"),
    device_id: str = Query(..., description="ID del dispositivo"),
):
    # Ajustar fechas para incluir todo el rango del día
    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d") + timedelta(
        hours=23, minutes=59, seconds=59
    )
    fecha_inicio_str = fecha_inicio_dt.strftime("%Y-%m-%dT%H:%M:%S")
    fecha_fin_str = fecha_fin_dt.strftime("%Y-%m-%dT%H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT fecha, filename, clase, probabilidad, image_url FROM analisis WHERE fecha BETWEEN ? AND ? AND device_id = ?",
        (fecha_inicio_str, fecha_fin_str, device_id),
    )
    rows = c.fetchall()
    conn.close()

    # Crear gráfico
    clases = [row[2] for row in rows]
    conteo = {clase: clases.count(clase) for clase in set(clases)}
    plt.figure(figsize=(6, 4))
    plt.bar(conteo.keys(), conteo.values(), color="skyblue")
    plt.title("Conteo de clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    unique_id = str(uuid.uuid4())
    grafico_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        f"grafico_{device_id}_{unique_id}.png",
    )
    plt.savefig(grafico_path)
    plt.close()

    # Crear PDF
    pdf_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        f"reporte_{device_id}_{unique_id}.pdf",
    )
    cpdf = canvas.Canvas(pdf_path, pagesize=letter)
    cpdf.setFont("Helvetica", 12)
    cpdf.drawString(30, 750, f"Reporte de análisis del {fecha_inicio} al {fecha_fin}")
    cpdf.drawImage(grafico_path, 30, 500, width=500, height=200)
    y = 480
    cpdf.drawString(
        30, y, "Fecha        Archivo        Clase        Probabilidad        Imagen"
    )
    y -= 20
    for row in rows:
        cpdf.drawString(
            30,
            y,
            f"{row[0][:19]}  {row[1][:15]}  {row[2]:<15}  {row[3]:.2f}  {row[4] if row[4] else ''}",
        )
        y -= 15
        if y < 50:
            cpdf.showPage()
            y = 750
    cpdf.save()
    # Enviar el PDF y luego eliminar los archivos temporales
    response = FileResponse(
        pdf_path, media_type="application/pdf", filename=f"reporte_{device_id}.pdf"
    )
    import threading
    import time

    def cleanup(paths):
        time.sleep(5)
        for p in paths:
            try:
                os.remove(p)
            except Exception:
                pass

    threading.Thread(
        target=cleanup, args=([pdf_path, grafico_path],), daemon=True
    ).start()
    return response


@app.get("/analisis/listado")
def listado_analisis(
    fecha_inicio: str = Query(..., description="Fecha inicio en formato YYYY-MM-DD"),
    fecha_fin: str = Query(..., description="Fecha fin en formato YYYY-MM-DD"),
    device_id: str = Query(..., description="ID del dispositivo"),
):
    # Ajustar fechas para incluir todo el rango del día
    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%Y-%m-%d") + timedelta(
        hours=23, minutes=59, seconds=59
    )
    fecha_inicio_str = fecha_inicio_dt.strftime("%Y-%m-%dT%H:%M:%S")
    fecha_fin_str = fecha_fin_dt.strftime("%Y-%m-%dT%H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, fecha, filename, clase, probabilidad, image_url FROM analisis WHERE fecha BETWEEN ? AND ? AND device_id = ? ORDER BY fecha DESC",
        (fecha_inicio_str, fecha_fin_str, device_id),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "fecha": row[1],
            "filename": row[2],
            "clase": row[3],
            "probabilidad": row[4],
            "image_url": row[5],
        }
        for row in rows
    ]
