# =====================
# import libraries
# =====================
import os
import json
import random
import gc
import time
import copy
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Image processing
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ML utilities
import timm
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.use("Agg")

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_PROP = fm.FontProperties(fname=FONT_PATH)


def normalize_str(s: str) -> str:
    """Normalize Unicode string to NFC for reliable comparison"""
    return unicodedata.normalize("NFC", s)

# =====================
# Font (Korean-safe, SSH/Agg compatible)
# =====================
matplotlib.use("Agg")  # 반드시 pyplot import 전에

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Korean font not found: {FONT_PATH}")

FONT_PROP = fm.FontProperties(fname=FONT_PATH)


# =====================
# Helper: Unicode-safe directory discovery
# =====================
def find_child_dir(parent: Path, prefix: str) -> Path:
    matches = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No directory starting with '{prefix}' under {parent}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple directories starting with '{prefix}' under {parent}")
    return matches[0]


# =====================
# CONFIG
# =====================
class CONFIG:
    seed = 42

    epochs = 10
    train_batch_size = 32
    valid_batch_size = 32
    img_size = [512, 512]

    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    is_pretrained = True

    learning_rate = 5e-5
    weight_decay = 1e-6
    min_lr = 1e-6

    n_risk_classes = 3
    risk_class_names = ['Low', 'Intermediate', 'High']

    project_root = Path("/home/snhyunchang/Mac_skin_cancer")

    # Dataset roots (Unicode-safe)
    dataset14_root = next(project_root.glob("14.*"))
    dataset15_root = next(project_root.glob("15.*"))

    # ---------- Dataset 14 ----------
    d14_level3 = find_child_dir(dataset14_root, "3.")
    d14_base   = find_child_dir(d14_level3, "1.")
    d14_train  = find_child_dir(d14_base, "Training")
    d14_val    = find_child_dir(d14_base, "Validation")

    dataset14_train_img   = find_child_dir(d14_train, "01")
    dataset14_train_label = find_child_dir(d14_train, "02")
    dataset14_val_img     = find_child_dir(d14_val, "01")
    dataset14_val_label   = find_child_dir(d14_val, "02")

    # ---------- Dataset 15 ----------
    d15_level3 = find_child_dir(dataset15_root, "3.")
    d15_base   = find_child_dir(d15_level3, "1.")
    d15_train  = find_child_dir(d15_base, "Training")
    d15_val    = find_child_dir(d15_base, "Validation")

    dataset15_train_img   = find_child_dir(d15_train, "01")
    dataset15_train_label = find_child_dir(d15_train, "02")
    dataset15_val_img     = find_child_dir(d15_val, "01")
    dataset15_val_label   = find_child_dir(d15_val, "02")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_workers = 4

    # Disease name mapping (unchanged)
    disease_name_mapping = {
        'TS_건선_정면': '건선', 'TS_건선_측면': '건선',
        'TS_아토피_정면': '아토피', 'TS_아토피_측면': '아토피',
        'TS_여드름_정면': '여드름', 'TS_여드름_측면': '여드름',
        'TS_정상_정면': '정상', 'TS_정상_측면': '정상',
        'TS_주사_정면': '주사', 'TS_주사_측면': '주사',
        'TS_지루_정면': '지루', 'TS_지루_측면': '지루',

        'VS_건선_정면': '건선', 'VS_건선_측면': '건선',
        'VS_아토피_정면': '아토피', 'VS_아토피_측면': '아토피',
        'VS_여드름_정면': '여드름', 'VS_여드름_측면': '여드름',
        'VS_정상_정면': '정상', 'VS_정상_측면': '정상',
        'VS_주사_정면': '주사', 'VS_주사_측면': '주사',
        'VS_지루_정면': '지루', 'VS_지루_측면': '지루',

        'TS_광선각화증': '광선각화증',
        'TS_기저세포암': '기저세포암',
        'TS_멜라닌세포모반': '멜라닌세포모반',
        'TS_보웬병': '보웬병',
        'TS_비립종': '비립종',
        'TS_사마귀': '사마귀',
        'TS_악성흑색종': '악성흑색종',
        'TS_지루각화증': '지루각화증',
        'TS_편평세포암': '편평세포암',
        'TS_표피낭종': '표피낭종',
        'TS_피부섬유종': '피부섬유종',
        'TS_피지샘증식증': '피지샘증식증',
        'TS_혈관종': '혈관종',
        'TS_화농 육아종': '화농육아종',
        'TS_흑색점': '흑색점',

        'VS_광선각화증': '광선각화증',
        'VS_기저세포암': '기저세포암',
        'VS_멜라닌세포모반': '멜라닌세포모반',
        'VS_보웬병': '보웬병',
        'VS_비립종': '비립종',
        'VS_사마귀': '사마귀',
        'VS_악성흑색종': '악성흑색종',
        'VS_지루각화증': '지루각화증',
        'VS_편평세포암': '편평세포암',
        'VS_표피낭종': '표피낭종',
        'VS_피부섬유종': '피부섬유종',
        'VS_피지샘증식증': '피지샘증식증',
        'VS_혈관종': '혈관종',
        'VS_화농 육아종': '화농육아종',
        'VS_흑색점': '흑색점',
    }

    risk_mapping = {
        '악성흑색종': 2, '기저세포암': 2, '편평세포암': 2,
        '보웬병': 1, '광선각화증': 1,
        '멜라닌세포모반': 0, '비립종': 0, '사마귀': 0,
        '지루각화증': 0, '표피낭종': 0, '피부섬유종': 0,
        '피지샘증식증': 0, '혈관종': 0, '화농육아종': 0,
        '흑색점': 0, '건선': 0, '아토피': 0, '여드름': 0,
        '주사': 0, '지루': 0, '정상': 0
    }


# =====================
# Seed
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG.seed)


print("dataset14_train_img:", CONFIG.dataset14_train_img)
print("dataset15_train_img:", CONFIG.dataset15_train_img)
print("Device:", CONFIG.device)
print("Risk Classes:", CONFIG.n_risk_classes)
print("Risk Levels:", CONFIG.risk_class_names)

# =====================
# Data collection, Dataset, Model, Training
# =====================
# (UNCHANGED — your existing logic continues here)
# ✅ From this point onward, your script runs exactly as intended.




# =====================
# Data path
# =====================

def collect_data_paths(img_root, label_root, dataset_name="Dataset"):
    """
    폴더를 스캔하여 (이미지 경로, JSON 경로, 라벨) 리스트 생성
    """

    data_list = []
    missing_folders = set()
    missing_json_count = 0

    if not img_root.exists():
        print(f" {dataset_name}: {img_root} 폴더가 없습니다.")
        return data_list

    try:
        folders = [p for p in img_root.iterdir() if p.is_dir()]
    except Exception as e:
        print(f" {dataset_name}: 폴더 목록 읽기 실패 - {e}")
        return data_list
    for folder_path in folders:
        folder = folder_path.name
        img_dir = folder_path

        # TS_ → TL_, VS_ → VL_
        label_folder = folder.replace('TS_', 'TL_').replace('VS_', 'VL_')
        label_dir = label_root / label_folder

        if not label_dir.exists():
            print(f" {folder}: 라벨 폴더 없음 ({label_folder})")
            continue

        try:
            png_files = [
                p for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            ]
        except Exception as e:
            print(f"{folder}: PNG 파일 목록 읽기 실패 - {e}")
            continue

        for png_path in png_files:
            png_file = png_path.name
            json_file = png_path.stem + ".json"
            json_path = label_dir / json_file

            if not json_path.exists():
                # 대소문자 다른 경우 fallback
                json_files = [
                    p for p in label_dir.iterdir()
                    if p.is_file() and p.name.lower() == json_file.lower()
                ]
                if not json_files:
                    missing_json_count += 1
                    print(f"[JSON 없음] {folder} / {png_file}")
                    continue
                json_path = json_files[0]

            normalized_folder = normalize_str(folder)
            disease_name = CONFIG.disease_name_mapping.get(normalized_folder)

            if disease_name is None:
                missing_folders.add(folder)
                continue

            data_list.append({
                "img_path": str(png_path.resolve()),
                "json_path": str(json_path.resolve()),
                "risk": CONFIG.risk_mapping[disease_name],
            })
    if len(missing_folders) > 0:
        print(f"\n[{dataset_name}] disease_name_mapping 누락 폴더:")
        for f in sorted(missing_folders):
            print(f"  - {f}")

    if missing_json_count > 0:
        print(f"[{dataset_name}] JSON 누락 이미지 수: {missing_json_count}")

    return data_list


# ============================================================
# CSV 저장 
# ============================================================
train_csv_path = CONFIG.project_root / "train_index2.csv"
valid_csv_path = CONFIG.project_root / "valid_index2.csv"

# 1. 파일이 이미 존재하는지 확인
if train_csv_path.exists() and valid_csv_path.exists():
    print("=" * 70)
    print("✅ 기존 CSV 파일이 발견되어 로드합니다. (데이터 수집 건너뜀)")
    print("=" * 70)

    train_df = pd.read_csv(train_csv_path, encoding="utf-8-sig")
    valid_df = pd.read_csv(valid_csv_path, encoding="utf-8-sig")

else:
    # 2. 파일이 없으면 새로 생성 (최초 1회 실행)
    print("=" * 70)
    print("⚠️ CSV 파일이 없습니다. 데이터를 새로 수집합니다...")
    print("=" * 70)

    train_data = []
    valid_data = []

    # ---------------- Training ----------------
    print("\n[Dataset 14 - Training]")
    train_data.extend(
        collect_data_paths(
            CONFIG.dataset14_train_img,
            CONFIG.dataset14_train_label,
            "Dataset 14 Training"
        )
    )

    print("\n[Dataset 15 - Training]")
    train_data.extend(
        collect_data_paths(
            CONFIG.dataset15_train_img,
            CONFIG.dataset15_train_label,
            "Dataset 15 Training"
        )
    )

    # ---------------- Validation ----------------
    print("\n[Dataset 14 - Validation]")
    valid_data.extend(
        collect_data_paths(
            CONFIG.dataset14_val_img,
            CONFIG.dataset14_val_label,
            "Dataset 14 Validation"
        )
    )

    print("\n[Dataset 15 - Validation]")
    valid_data.extend(
        collect_data_paths(
            CONFIG.dataset15_val_img,
            CONFIG.dataset15_val_label,
            "Dataset 15 Validation"
        )
    )

    print(f"\nTraining 샘플 수   : {len(train_data):,}")
    print(f"Validation 샘플 수 : {len(valid_data):,}")

    if len(train_data) == 0:
        raise RuntimeError("Training 데이터가 0개입니다. 경로를 확인하세요.")

    # DataFrame 생성 및 저장
    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_csv_path, index=False, encoding="utf-8-sig")

    print(
        f"\n✅ CSV 저장 완료:\n"
        f" - {train_csv_path}\n"
        f" - {valid_csv_path}"
    )


# ============================================================
# Validation → Validation / Test split (Pathlib 기반)
# ============================================================

VAL_SPLIT_PATH = CONFIG.project_root / "val_index.csv"
TEST_SPLIT_PATH = CONFIG.project_root / "test_index.csv"

if not TEST_SPLIT_PATH.exists():
    # 최초 1회만 split 수행
    from sklearn.model_selection import train_test_split

    val_df, test_df = train_test_split(
    valid_df,
    test_size=0.5,
    random_state=CONFIG.seed,
    stratify=valid_df["risk"]
)


    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    val_df.to_csv(VAL_SPLIT_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_SPLIT_PATH, index=False, encoding="utf-8-sig")

    print("✅ Validation/Test split 최초 1회 수행 및 저장 완료")
    print(f"   ├─ Validation: {len(val_df):,}")
    print(f"   └─ Test       : {len(test_df):,}")

else:
    # 이후 모든 실험은 동일 split 로드
    val_df = pd.read_csv(VAL_SPLIT_PATH, encoding="utf-8-sig")
    test_df = pd.read_csv(TEST_SPLIT_PATH, encoding="utf-8-sig")

print(f"Train: {len(train_df)}, Valid: {len(val_df)}, Test: {len(test_df)}")



# =========================================================
# Dataset
# =========================================================
def transform_train(img):
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.7
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(std_range=(0.02, 0.05))
        ], p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5
        ),
        A.RandomShadow(p=0.3),

        A.Normalize(),
        ToTensorV2(),
    ])(image=img)["image"]


def transform_val(img):
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2(),
    ])(image=img)["image"]
    
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["img_path"]).convert("RGB")
        risk_label = torch.tensor(row["risk"], dtype=torch.long)

        bbox = None
        x, y, w, h = 0, 0, img.width, img.height

        try:
            with open(row["json_path"], 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            annotations = json_data.get("annotations", [])
            if isinstance(annotations, list) and len(annotations) > 0:
                bbox = annotations[0].get("bbox", None)

            if isinstance(bbox, dict):
                x = int(bbox.get("xpos", 0))
                y = int(bbox.get("ypos", 0))
                w = int(bbox.get("width", img.width))
                h = int(bbox.get("height", img.height))

        except Exception as e:
            # JSON 실패 시 전체 이미지 사용 (fallback)
            print(f"⚠️ JSON 읽기 실패: {row['json_path']} - {e}")

    # Safe crop
        if w < img.width or h < img.height:
            x = max(0, x)
            y = max(0, y)
            x2 = min(x + w, img.width)
            y2 = min(y + h, img.height)
            if x2 > x and y2 > y:
                img = img.crop((x, y, x2, y2))

    # Transform
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        risk_label = torch.tensor(row["risk"], dtype=torch.long)

        return img, risk_label

test_dataset = SkinDataset(test_df, transform=transform_val)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG.valid_batch_size,
    shuffle=False,
    num_workers=CONFIG.n_workers,
    pin_memory=True
)

print(f"✅ Test DataLoader: {len(test_loader)} batches")
    
train_dataset = SkinDataset(train_df, transform=transform_train)
valid_dataset = SkinDataset(val_df, transform=transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG.train_batch_size,
    shuffle=True,
    num_workers=CONFIG.n_workers,
    pin_memory=True,
    drop_last=False
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG.valid_batch_size,
    shuffle=False,
    num_workers=CONFIG.n_workers,
    pin_memory=True
)
print(f"✅ Train DataLoader: {len(train_loader)} batches")
print(f"✅ Valid DataLoader: {len(valid_loader)} batches")

class SkinDiseaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            CONFIG.model_name,
            pretrained=CONFIG.is_pretrained,
            num_classes=0,
            global_pool="avg"
        )

        in_features = self.backbone.num_features

        self.risk_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, CONFIG.n_risk_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        risk_logits = self.risk_head(feats)
        return risk_logits


def freeze_backbone(model):
    for name, param in model.backbone.named_parameters():
        if "blocks.5" not in name and "blocks.6" not in name:
            param.requires_grad = False


criterion_risk = nn.CrossEntropyLoss()

# =======================================
# Train one epoch
# =======================================

def train_one_epoch(model, optimizer, train_loader, epoch):
    model.train()

    RISK_LOSS_WEIGHT = min(1.0, epoch / 10)

    y_preds, y_trues = [], []
    running_loss, dataset_size = 0.0, 0

    bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")

    for images, risk_labels in bar:
        bs = images.size(0)

        images = images.to(CONFIG.device, dtype=torch.float)
        risk_labels = risk_labels.to(CONFIG.device)

        optimizer.zero_grad()
        risk_logits = model(images)

        loss_r = criterion_risk(risk_logits, risk_labels)

        loss = RISK_LOSS_WEIGHT * loss_r
        loss.backward()
        optimizer.step()

        preds = risk_logits.argmax(dim=1)
        y_preds.append(preds.detach().cpu().numpy())
        y_trues.append(risk_labels.detach().cpu().numpy())

        running_loss += loss.item() * bs
        dataset_size += bs

        epoch_loss = running_loss / dataset_size
        epoch_acc = (preds.cpu().numpy() == risk_labels.cpu().numpy()).mean()
        bar.set_postfix(
            Loss=f"{epoch_loss:.4f}",
            Acc=f"{epoch_acc:.4f}",
            LR=optimizer.param_groups[0]["lr"]
        )

    return epoch_loss, epoch_acc

# ===========================================================
# Validation one epoch
# ===========================================================

@torch.inference_mode()
def valid_one_epoch(model, valid_loader, epoch):
    model.eval()
    
    RISK_LOSS_WEIGHT = min(1.0, epoch / 10)

    y_preds, y_trues = [], []
    running_loss, dataset_size = 0.0, 0

    bar = tqdm(valid_loader, desc=f"Valid Epoch {epoch}")

    for images, risk_labels in bar:
        bs = images.size(0)

        images = images.to(CONFIG.device, dtype=torch.float)
        risk_labels = risk_labels.to(CONFIG.device)

        risk_logits = model(images)

        loss_r = criterion_risk(risk_logits, risk_labels)

        loss = RISK_LOSS_WEIGHT * loss_r

        preds = risk_logits.argmax(dim=1)
        y_preds.append(preds.cpu().numpy())
        y_trues.append(risk_labels.cpu().numpy())

        running_loss += loss.item() * bs
        dataset_size += bs

        epoch_loss = running_loss / dataset_size
        epoch_acc = (np.concatenate(y_preds) == np.concatenate(y_trues)).mean()

        bar.set_postfix(Loss=f"{epoch_loss:.4f}", Acc=f"{epoch_acc:.4f}")

    return epoch_loss, epoch_acc
# ==========================
# Early Stopping
# ==========================
class EarlyStopping:
    def __init__(self, patience=3, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ===============================================================
# Training Runner
# ==============================================================
def run_training(
    model,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    num_epochs=CONFIG.epochs
):
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}")
    
    start_time = time.time()
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = defaultdict(list)
    
    # ✅ Early Stopping 초기화
    early_stopper = EarlyStopping(patience=3)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        
        # -------------------------
        # Train
        # -------------------------
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, epoch
        )
        
        # -------------------------
        # Validation
        # -------------------------
        valid_loss, valid_acc = valid_one_epoch(
            model, valid_loader, epoch
        )

        # -------------------------
        # History 기록
        # -------------------------
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        if scheduler is not None:
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        # -------------------------
        # Logging
        # -------------------------
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}"
        )

        # -------------------------
        # Best model 저장
        # -------------------------
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
            save_path = (
            CONFIG.project_root
            / f"best_model_epoch{epoch}_acc{best_acc:.4f}.bin"
            )

            torch.save(model.state_dict(), save_path)
            print(f"Best model saved: {save_path}")

        # -------------------------
        # ✅ Early Stopping 판단 (scheduler 이전!)
        # -------------------------
        early_stopper.step(valid_acc)
        if early_stopper.early_stop:
            print("🛑 Early stopping triggered")
            break

        # -------------------------
        # Scheduler step (epoch 단위)
        # -------------------------
        if scheduler is not None:
            scheduler.step()

    # -------------------------
    # Training 종료 요약
    # -------------------------
    elapsed = time.time() - start_time
    print(
        f"\n{'='*70}\n"
        f"Training finished in "
        f"{elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s\n"
        f"Best Validation Accuracy: {best_acc:.4f}\n"
        f"{'='*70}"
    )
    
    model.load_state_dict(best_model_wts)
    return model, history

@torch.inference_mode()
def comprehensive_evaluation(model, data_loader, dataset_name="Test"):
    model.eval()

    all_preds, all_trues = [], []

    for images, risk_labels in tqdm(data_loader, desc=f"{dataset_name} Evaluation"):
        images = images.to(CONFIG.device)
        risk_labels = risk_labels.to(CONFIG.device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_trues.append(risk_labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    acc = accuracy_score(y_true, y_pred)

    print(f"\n📊 {dataset_name} Risk Stratification")
    print(f"{'='*60}")
    print(f"Overall Risk Accuracy : {acc:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=CONFIG.risk_class_names,
        digits=4
    ))

    return {
        "risk_accuracy": acc,
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


def visualize_model_performance(history, results=None, save_dir=None):
    """
    모델 학습 및 평가 결과를 종합적으로 시각화
    
    Args:
        history: 학습 이력 (dict)
        results: comprehensive_evaluation 결과 (dict, optional)
        save_dir: 저장 경로 (기본값: CONFIG.project_root)
    """
    if save_dir is None:
        save_dir = CONFIG.project_root
    if history is None:
        print("⚠️ No training history provided — skipping training curves.")
        return
    save_dir = Path(save_dir)  # Path 객체로 변환
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 학습 곡선 (Loss & Accuracy)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Performance Dashboard', fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1-1. Loss 곡선
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['valid_loss'], 'r-s', label='Valid Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontproperties=FONT_PROP)
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontproperties=FONT_PROP)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1-2. Accuracy 곡선
    axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['valid_acc'], 'r-s', label='Valid Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # 1-3. Learning Rate 스케줄
    if 'learning_rate' in history:
        axes[1, 0].plot(epochs, history['learning_rate'], 'g-^', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # 1-4. Overfitting 분석
    train_acc = np.array(history['train_acc'])
    valid_acc = np.array(history['valid_acc'])
    gap = train_acc - valid_acc
    
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Train Acc - Valid Acc', fontsize=12)
    axes[1, 1].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved: {save_path}")
    plt.close()
        
    # ========================================
    # 3. 위험도별 성능 (Risk-Level)
    # ========================================
    if 'risk_sensitivities' in results and 'risk_fnrs' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Risk-Level Performance', fontsize=16, fontweight='bold')
            
        risk_names = ['Low', 'Intermediate', 'High']
        risk_colors = ['green', 'orange', 'red']
            
        sensitivities = [
            results['risk_sensitivities']['Low'],
            results['risk_sensitivities']['Intermediate'],
            results['risk_sensitivities']['High']
        ]
        fnrs = [
            results['risk_fnrs']['Low'],
            results['risk_fnrs']['Intermediate'],
            results['risk_fnrs']['High']
        ]
            
        # 3-1. Sensitivity
        bars = axes[0].bar(risk_names, sensitivities, color=risk_colors, alpha=0.7, edgecolor='black')
        axes[0].set_ylabel('Sensitivity (Recall)', fontsize=12, fontproperties=FONT_PROP)
        axes[0].set_title('Sensitivity by Risk Level', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
            
        # 값 표시
        for bar, val in zip(bars, sensitivities):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        # 3-2. False Negative Rate
        bars = axes[1].bar(risk_names, fnrs, color=risk_colors, alpha=0.7, edgecolor='black')
        axes[1].set_ylabel('False Negative Rate', fontsize=12, fontproperties=FONT_PROP)
        axes[1].set_title('FNR by Risk Level (Lower is Better)', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
            
        # 값 표시
        for bar, val in zip(bars, fnrs):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        save_path = save_dir / 'risk_level_performance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Risk-level performance saved: {save_path}")
        plt.close()
        



if __name__ == "__main__":
    # 1. 모델 초기화
    model = SkinDiseaseModel().to(CONFIG.device)
    
    freeze_backbone(model)

    # 2. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG.learning_rate, 
        weight_decay=CONFIG.weight_decay
    )
    
    # 3. Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.epochs,
        eta_min=CONFIG.min_lr
    )

    # 4. 학습 실행
    model, history = run_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=CONFIG.epochs
    )

    best_ckpt = CONFIG.project_root / "best_model_epoch3_acc0.9985.bin"
    model.load_state_dict(torch.load(best_ckpt, map_location=CONFIG.device))
    model.eval()



    # ✅ 5. Validation 평가
    print("\n📊 Final Validation Evaluation (Best Model)")
    val_results = comprehensive_evaluation(
        model=model,
        data_loader=valid_loader,
        dataset_name="Validation (Best Model)"
    )
    
    # ✅ 6. 시각화 생성 (학습 곡선 + Validation 결과)
    print("\n🎨 Generating visualizations...")
    visualize_model_performance(
        history=history,
        results=val_results,
        save_dir=CONFIG.project_root
    )
    
    # 7. Test Evaluation (옵션)
    RUN_TEST = True
    
    if RUN_TEST:
        print("\n🚀 Test set evaluation started")
        
        test_results = comprehensive_evaluation(
            model=model,
            data_loader=test_loader,
            dataset_name="Test"
        )
        test_save_dir = CONFIG.project_root / "test_results"
        test_save_dir.mkdir(parents=True, exist_ok=True)

        # Test 결과 시각화
        print("\n🎨 Generating test visualizations...")
        visualize_model_performance(
            history=history,
            results=test_results,
            save_dir=test_save_dir
        )
        
        print("\n" + "="*70)
        print("Key Performance Indicators")
        print("="*70)
        print(f"  Risk Accuracy : {test_results['risk_accuracy']:.4f}")
        print("="*70)

    # 8. Training history 저장
    history_df = pd.DataFrame(history)
    history_path = CONFIG.project_root / "training_history2.csv"
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f"📁 Training history saved: {history_path}")