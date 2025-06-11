# Pluton API

API para clasificación de frutas frescas y podridas usando Deep Learning, con generación de reportes y almacenamiento en la nube.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Configuración](#configuración)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Uso de la API](#uso-de-la-api)
  - [Endpoints](#endpoints)
- [Despliegue con Docker](#despliegue-con-docker)
- [Notas y recomendaciones](#notas-y-recomendaciones)

---

## Descripción

Este proyecto implementa una API REST con FastAPI para la clasificación de imágenes de frutas (plátano, mango y naranja, frescos y podridos) usando un modelo CNN entrenado en PyTorch. Permite:

- Subir imágenes y obtener la predicción de clase y probabilidad.
- Almacenar resultados y metadatos en una base de datos SQLite.
- Subir imágenes a Cloudinary.
- Generar reportes en PDF con gráficos de análisis.
- Consultar el historial de análisis por dispositivo y rango de fechas.

---

## Estructura del Proyecto

```
.
├── app/
│   ├── main.py              # Código principal de la API
│   ├── model_loader.py      # Carga y predicción del modelo
│   └── __init__.py
├── models/
│   ├── modelo_frutas.pth    # Modelo entrenado
│   ├── analisis.db          # Base de datos SQLite
│   └── .gitkeep
├── notebook-train.ipynb     # Notebook de entrenamiento
├── requirements.txt         # Dependencias Python
├── Dockerfile               # Imagen Docker
├── docker-compose.yml       # Orquestación Docker
├── .gitignore
└── ...
```

---

## Requisitos

- Python 3.8+
- Docker (opcional, recomendado para despliegue)
- Cuenta en [Cloudinary](https://cloudinary.com/) para almacenamiento de imágenes

### Dependencias principales

- fastapi
- uvicorn
- torch
- torchvision
- pillow
- reportlab
- matplotlib
- cloudinary
- python-dotenv

Ver el archivo `requirements.txt` para la lista completa.

---

## Configuración

1. **Variables de entorno**

Crea un archivo `.env` en la raíz del proyecto con el siguiente contenido (ajusta los valores):

```
CLOUDINARY_CLOUD_NAME=tu_cloud_name
CLOUDINARY_API_KEY=tu_api_key
CLOUDINARY_API_SECRET=tu_api_secret
DB_PATH=./models/analisis.db
```

2. **Modelos y base de datos**

- El modelo entrenado debe estar en `models/modelo_frutas.pth`.
- La base de datos SQLite se crea automáticamente en `models/analisis.db` al iniciar la API.

---

## Entrenamiento del Modelo

El archivo `notebook-train.ipynb` contiene el flujo completo para:

- Descargar datasets de Kaggle (bananas, naranjas y mangos, frescos y podridos).
- Reorganizar los datos.
- Definir y entrenar una CNN simple en PyTorch.
- Guardar el modelo entrenado como `modelo_frutas.pth`.

Puedes modificar el notebook para ajustar hiperparámetros o clases.

---

## Uso de la API

### Ejecución local

```bash
uvicorn app.main:app --reload
```

La API estará disponible en `http://localhost:8000`.

### Endpoints

#### 1. `POST /predict/`

- **Descripción:** Clasifica una imagen de fruta.
- **Parámetros:**
  - `file`: Imagen a clasificar (formato multipart/form-data)
  - `device_id`: ID del dispositivo (query)
- **Respuesta:**
  ```json
  {
  	"prediction": "freshbanana",
  	"probabilidad": 0.98,
  	"image_url": "https://res.cloudinary.com/..."
  }
  ```

#### 2. `GET /reporte/pdf`

- **Descripción:** Genera un PDF con el resumen de análisis en un rango de fechas para un dispositivo.
- **Parámetros:**
  - `fecha_inicio`: Fecha inicio (YYYY-MM-DD)
  - `fecha_fin`: Fecha fin (YYYY-MM-DD)
  - `device_id`: ID del dispositivo
- **Respuesta:** PDF descargable con gráfico y tabla de análisis.

#### 3. `GET /analisis/listado`

- **Descripción:** Lista los análisis realizados en un rango de fechas para un dispositivo.
- **Parámetros:**
  - `fecha_inicio`: Fecha inicio (YYYY-MM-DD)
  - `fecha_fin`: Fecha fin (YYYY-MM-DD)
  - `device_id`: ID del dispositivo
- **Respuesta:** Lista de análisis en formato JSON.

---

## Despliegue con Docker

1. **Construir y levantar los servicios:**

```bash
docker-compose up --build
```

2. **Acceso a la API:**

La API estará disponible en `http://localhost:8000`.

- El contenedor monta la carpeta `models/` y el archivo `.env` para persistencia y configuración.

---

## Notas y recomendaciones

- **Cloudinary:** Es necesario tener una cuenta y configurar las variables de entorno para el almacenamiento de imágenes.
- **Base de datos:** Por defecto se usa SQLite, pero puedes adaptar el código para otros motores si lo requieres.
- **Seguridad:** No subas tu archivo `.env` ni la base de datos a repositorios públicos.
- **Extensión:** Puedes adaptar el modelo y la API para otras clases de frutas o productos.
