import torch
import torch.nn as nn
from backend.config import ACTION_SIZE, DEVICE

class ChessPolicyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(12,64, kernel_size =3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(128,128, kernel_size = 3, padding = 1),
            nn.ReLU()
            )

        # conv output: (batch, 128, 8, 8) -> flattened = 128*8*8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, ACTION_SIZE)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        logits = self.fc(x)
        return logits
    


def load_model(model_path: str = None) -> ChessPolicyNet:
    model = ChessPolicyNet().to(DEVICE)

    if model_path is not None:
        try:
            model.load_state_dict(torch.load(model_path, map_location= DEVICE))
            print(f"Loaded Model from {model_path}")
        except:
            print("File Not Found, Starting Fresh")

    return model

