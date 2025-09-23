import os, glob, pickle, json
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from catboost import CatBoostClassifier

# ---------------- CONFIG ---------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
IMG_SIZE = 224
NUM_SLICES = 32
SLICE_SPREAD = 0.6
MIN_NONZERO_FRAC = 0.002

# ---------------- UTILITIES ---------------- #
def load_nifti(path):
    img = nib.load(path)
    arr = img.get_fdata().astype(np.float32)
    nz = arr[arr > 0]
    if nz.size > 50:
        m, s = nz.mean(), nz.std() + 1e-6
        arr = (arr - m) / s
    arr = np.clip(arr, -3.0, 3.0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def pick_slices(volume, num_slices=32, spread=0.6):
    Z = volume.shape[-1]
    z0 = int((1.0 - spread)/2 * Z)
    z1 = int((1.0 + spread)/2 * Z)
    z_idx = np.linspace(z0, max(z0+1, z1-1), num_slices, dtype=int)

    kept = []
    for z in z_idx:
        sl = volume[..., z]
        if (sl > 0).mean() >= MIN_NONZERO_FRAC:
            kept.append(z)
    if len(kept) == 0:
        kept = [Z//2]
    return kept

# ---------------- FEATURE EXTRACTORS ---------------- #
resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

# SWIN Transformer
swin = models.swin_v2_t(weights='DEFAULT')
swin_feat = create_feature_extractor(swin, return_nodes={'flatten': 'feat'}).to(device).eval()

# MobileNetV3
mnet = models.mobilenet_v3_large(weights='DEFAULT')
mnet_feat = create_feature_extractor(mnet, return_nodes={'features.15': 'feat_last'}).to(device).eval()

gap = nn.AdaptiveAvgPool2d((1,1)).to(device)

@torch.no_grad()
def extract_volume_features(vol_path):
    vol = load_nifti(vol_path)
    z_list = pick_slices(vol, NUM_SLICES, SLICE_SPREAD)

    swin_feats, mnet_feats = [], []
    for z in z_list:
        sl = vol[..., z]
        t = resize(sl)  # (1,224,224)
        t = t.repeat(3,1,1).unsqueeze(0).to(device)  # (1,3,224,224)

        # SWIN features
        out_s = swin_feat(t)['feat']
        if out_s.ndim > 2:
            out_s = torch.flatten(out_s, 1)
        swin_feats.append(out_s.squeeze(0).cpu())

        # MobileNet features
        fm = mnet_feat(t)['feat_last']
        pooled = gap(fm).view(1, -1)
        mnet_feats.append(pooled.squeeze(0).cpu())

    swin_vec = torch.stack(swin_feats, 0).mean(0)
    mnet_vec = torch.stack(mnet_feats, 0).mean(0)
    fused = torch.cat([swin_vec, mnet_vec], dim=0).numpy()
    return fused

# ---------------- LOAD MODELS ---------------- #
def load_models(load_dir):
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Model directory not found: {load_dir}")

    meta_path = os.path.join(load_dir, "meta.json")
    xgb_path = os.path.join(load_dir, "xgb_model.pkl")
    cat_path = os.path.join(load_dir, "cat_model.cbm")
    rf_path = os.path.join(load_dir, "rf_meta.pkl")

    for p in [meta_path, xgb_path, cat_path, rf_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file missing: {p}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)

    cat_model = CatBoostClassifier()
    cat_model.load_model(cat_path)

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    return xgb_model, cat_model, rf_model, meta

SAVE_DIR = "./models/mri_model"
xgb, cat, rf, meta = load_models(SAVE_DIR)

# ---------------- PREDICTION ---------------- #
def predict_volume(path):
    f = extract_volume_features(path).reshape(1,-1)

    p_xgb = xgb.predict_proba(f)[:,1].reshape(-1,1)
    p_cat = cat.predict_proba(f)[:,1].reshape(-1,1)

    f_meta = np.concatenate([f, p_xgb, p_cat], axis=1)
    p = rf.predict_proba(f_meta)[:,1][0]

    label = "MS" if p >= 0.5 else "NORMAL"
    fname = os.path.basename(path)
    print(f"{fname:25s} -> Probability (MS) = {p:.4f} | Predicted: {label}")
    return fname, label, float(p)

def predict_folder(folder, threshold=0.5):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(glob.glob(os.path.join(folder, "*.nii*")))
    if len(files) == 0:
        print(f"No NIfTI files found in {folder}")
        return []

    results = []
    for f in files:
        fname, label, prob = predict_volume(f)
        results.append((fname, label, prob))
    return results

# ---------------- USAGE ---------------- #
predict_folder("./test/mri", threshold=0.5)
