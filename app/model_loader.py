# /app/model_loader.py
import torch
import os
from torch import nn


# Definición del modelo (arquitectura)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 56 * 56, 56)
        self.fc2 = nn.Linear(56, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(self.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(self.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(
    model_path: str = os.path.join(
        os.path.dirname(__file__), "..", "models", "modelo_frutas.pth"
    )
):
    try:
        model = Net()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        print(f"Modelo cargado correctamente desde: {model_path}")
        print("Arquitectura del modelo:", model)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {str(e)}")


def predict(model: nn.Module, tensor: torch.Tensor) -> str:
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(probabilities, 1)
        class_index = preds.item()
        print(f"Probabilidades: {probabilities.data.numpy()[0].round(2)}")
        return "Banana fresca 🍌" if class_index == 0 else "Banana podrida 🤢"
