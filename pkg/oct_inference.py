import os
import torch
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- CONFIG ---------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "tf_efficientnet_b4_ns"
CKPT_PATH = "./models/oct_model/oct_model.pth"
IMAGE_SIZE = 300

# ---------------- PREPROCESSING ---------------- #
transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ---------------- LOAD MODELS ---------------- #
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.to(DEVICE)
model.eval()

# ---------------- PREDICTION ---------------- #
def predict_image(image_path):
    """Predict probability of MS from an OCT slice."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    img = transform(image=img)["image"]
    img = img.unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logit = model(img).squeeze(1)
        prob = torch.sigmoid(logit).item()

    pred_class = 1 if prob >= 0.5 else 0  # 1 = MS, 0 = Healthy
    print(f"[{os.path.basename(image_path)}] MS probability: {prob:.4f}, Prediction: {'MS' if pred_class==1 else 'Healthy'}")
    return prob, pred_class

def predict_folder(folder_path, extensions=(".png", ".jpg", ".jpeg")):
    """Predict MS for all images in a folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    results = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(extensions):
            img_path = os.path.join(folder_path, file_name)
            try:
                prob, pred = predict_image(img_path)
                results[file_name] = {"probability": prob, "prediction": pred}
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

    if not results:
        print("No valid images found in folder.")
    return results

# ---------------- USAGE ---------------- #
# predictions = predict_folder("./test/oct")