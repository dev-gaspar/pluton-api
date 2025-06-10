# /app/model_loader.py
import torch
import os
import torch.nn.functional as F
from torch import nn


# Definición del modelo (arquitectura)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 6)  # 6 clases

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(
    model_path: str = os.path.join(
        os.path.dirname(__file__), "..", "models", "modelo_frutas.pth"
    )
):
    try:
        model = SimpleCNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        print(f"Modelo cargado correctamente desde: {model_path}")
        print("Arquitectura del modelo:", model)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {str(e)}")


# Lista de nombres de clases (ajusta si tu modelo fue entrenado con otros nombres)
CLASS_NAMES = [
    "freshbanana",
    "freshmango",
    "freshoranges",
    "rottenbanana",
    "rottenmango",
    "rottenoranges",
]


def predict(model: nn.Module, tensor: torch.Tensor) -> str:
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(probabilities, 1)
        class_index = preds.item()
        print(f"Probabilidades: {probabilities.data.numpy()[0].round(2)}")
        return CLASS_NAMES[class_index]
