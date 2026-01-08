import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# CONFIG (DO NOT CHANGE PATHS)
# -----------------------------
PKG_DIR = "pkg/oct_validation"
MODEL_PATH = os.path.join(PKG_DIR, "oct_valid.pt")
CENTER_PATH = os.path.join(PKG_DIR, "center.pt")
THRESHOLD_PATH = os.path.join(PKG_DIR, "threshold.txt")

LATENT_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
class OneClassEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(pretrained=False)
        self.feature_extractor = nn.Sequential(
            *list(base.children())[:-1]
        )
        self.embedding = nn.Linear(1280, LATENT_DIM)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = torch.flatten(x, start_dim=1)
        return self.embedding(x)

# -----------------------------
# LOAD ASSETS (ONCE)
# -----------------------------
_model = None
_center = None
_threshold = None

def _load_assets():
    global _model, _center, _threshold

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("oct_valid.pt not found")

    if not os.path.exists(CENTER_PATH):
        raise FileNotFoundError("center.pt not found")

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError("threshold.txt not found")

    _model = OneClassEfficientNet().to(device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _model.eval()

    _center = torch.load(CENTER_PATH, map_location=device)

    with open(THRESHOLD_PATH, "r") as f:
        _threshold = float(f.read().strip())

# -----------------------------
# PREPROCESS
# -----------------------------
_transform = transforms.Compose([
    transforms.ToTensor()
])

def _load_image(image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found")

    img = Image.open(image_path).convert("RGB")
    img = _transform(img).unsqueeze(0)  # [1, C, H, W]
    return img.to(device)

# -----------------------------
# PUBLIC API (THIS IS WHAT YOU USE)
# -----------------------------
def validate_oct(image_path: str):
    """
    Validate a single image.

    Returns:
        is_oct (bool)
        score (float)
        threshold (float)
    """
    _load_assets()
    img = _load_image(image_path)

    with torch.no_grad():
        emb = _model(img)
        distance = torch.norm(emb - _center, dim=1).item()

    is_oct = distance <= _threshold
    return is_oct, distance, _threshold
