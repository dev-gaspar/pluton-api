# Dockerfile para banana-checker-api
FROM python:3.11-slim

# Variables de entorno para evitar buffering y asegurar logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto por defecto de FastAPI
EXPOSE 8000

# Comando para lanzar el backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 