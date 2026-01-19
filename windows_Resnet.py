# =====================
# import libraries
# =====================
# Standard libraries
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path

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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import gc
import time
import copy
from collections import defaultdict


#한글폰트
# Linux 한글 폰트 설정
font_candidates = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
]

for font_path in font_candidates:
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
        break

plt.rcParams['axes.unicode_minus'] = False

# ==================
# config
# ==================
class CONFIG:
    # Random seed
    seed = 42
    
    # Training hyperparameters
    epochs = 10
    train_batch_size = 32
    valid_batch_size = 32
    img_size = [512, 512]
    
    # Model settings
    n_classes = 21
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    model_name_resnet = "resnet50"
    is_pretrained = True
    
    # Optimizer & Scheduler
    learning_rate = 5e-5
    weight_decay = 1e-6
    min_lr = 1e-6
    scheduler = "CosineAnnealingLR"
    
    # Data paths
    project_root = r"C:\Users\Donghyeok Choi\Dropbox\피부암 project"
    
    dataset14_root = os.path.join(
        project_root, "14.안면부 피부질환 이미지 합성데이터", "3.개방데이터", "1.데이터"
    )
    dataset14_train_img = os.path.join(dataset14_root, "Training", "01.원천데이터")
    dataset14_train_label = os.path.join(dataset14_root, "Training", "02.라벨링데이터")
    dataset14_val_img = os.path.join(dataset14_root, "Validation", "01.원천데이터")
    dataset14_val_label = os.path.join(dataset14_root, "Validation", "02.라벨링데이터")
    
    dataset15_root = os.path.join(
        project_root, "15.피부종양 이미지 합성데이터", "3.개방데이터", "1.데이터"
    )
    dataset15_train_img = os.path.join(dataset15_root, "Training", "01.원천데이터")
    dataset15_train_label = os.path.join(dataset15_root, "Training", "02.라벨링데이터")
    dataset15_val_img = os.path.join(dataset15_root, "Validation", "01.원천데이터")
    dataset15_val_label = os.path.join(dataset15_root, "Validation", "02.라벨링데이터")
    
    # Hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_workers = 4
    
    # ✅ 질환명 정규화 (Training + Validation)
    disease_name_mapping = {
        # Training - Dataset 14
        'TS_건선_정면': '건선', 'TS_건선_측면': '건선',
        'TS_아토피_정면': '아토피', 'TS_아토피_측면': '아토피',
        'TS_여드름_정면': '여드름', 'TS_여드름_측면': '여드름',
        'TS_정상_정면': '정상', 'TS_정상_측면': '정상',
        'TS_주사_정면': '주사', 'TS_주사_측면': '주사',
        'TS_지루_정면': '지루', 'TS_지루_측면': '지루',
        
        # Validation - Dataset 14
        'VS_건선_정면': '건선', 'VS_건선_측면': '건선',
        'VS_아토피_정면': '아토피', 'VS_아토피_측면': '아토피',
        'VS_여드름_정면': '여드름', 'VS_여드름_측면': '여드름',
        'VS_정상_정면': '정상', 'VS_정상_측면': '정상',
        'VS_주사_정면': '주사', 'VS_주사_측면': '주사',
        'VS_지루_정면': '지루', 'VS_지루_측면': '지루',
        
        # Training - Dataset 15
        'TS_광선각화증': '광선각화증', 'TS_기저세포암': '기저세포암',
        'TS_멜라닌세포모반': '멜라닌세포모반', 'TS_보웬병': '보웬병',
        'TS_비립종': '비립종', 'TS_사마귀': '사마귀',
        'TS_악성흑색종': '악성흑색종', 'TS_지루각화증': '지루각화증',
        'TS_편평세포암': '편평세포암', 'TS_표피낭종': '표피낭종',
        'TS_피부섬유종': '피부섬유종', 'TS_피지샘증식증': '피지샘증식증',
        'TS_혈관종': '혈관종', 'TS_화농 육아종': '화농육아종',
        'TS_흑색점': '흑색점',
        
        # Validation - Dataset 15
        'VS_광선각화증': '광선각화증', 'VS_기저세포암': '기저세포암',
        'VS_멜라닌세포모반': '멜라닌세포모반', 'VS_보웬병': '보웬병',
        'VS_비립종': '비립종', 'VS_사마귀': '사마귀',
        'VS_악성흑색종': '악성흑색종', 'VS_지루각화증': '지루각화증',
        'VS_편평세포암': '편평세포암', 'VS_표피낭종': '표피낭종',
        'VS_피부섬유종': '피부섬유종', 'VS_피지샘증식증': '피지샘증식증',
        'VS_혈관종': '혈관종', 'VS_화농 육아종': '화농육아종',
        'VS_흑색점': '흑색점',
    }
    
    # Risk mapping
    risk_mapping = {
        '악성흑색종': 2, '기저세포암': 2, '편평세포암': 2,
        '보웬병': 1, '광선각화증': 1,
        '멜라닌세포모반': 0, '비립종': 0, '사마귀': 0, '지루각화증': 0,
        '표피낭종': 0, '피부섬유종': 0, '피지샘증식증': 0, '혈관종': 0,
        '화농육아종': 0, '흑색점': 0,
        '건선': 0, '아토피': 0, '여드름': 0, '주사': 0, '지루': 0, '정상': 0
    }
    
    class_names = sorted(risk_mapping.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"Device: {CONFIG.device}")
print(f"Classes: {CONFIG.n_classes}")

# ===================
# Set Seed
#===================

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG.seed)




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


    if not os.path.exists(img_root):
        print(f" {dataset_name}: {img_root} 폴더가 없습니다.")
        return data_list

    try:
        folders = [
            f for f in os.listdir(img_root)
            if os.path.isdir(os.path.join(img_root, f))
        ]
    except Exception as e:
        print(f" {dataset_name}: 폴더 목록 읽기 실패 - {e}")
        return data_list

    for folder in folders:
        img_dir = os.path.join(img_root, folder)

        # TS_ → TL_, VS_ → VL_
        label_folder = folder.replace('TS_', 'TL_').replace('VS_', 'VL_')
        label_dir = os.path.join(label_root, label_folder)

        if not os.path.exists(label_dir):
            print(f" {folder}: 라벨 폴더 없음 ({label_folder})")
            continue

        try:
            png_files = [
                f for f in os.listdir(img_dir)
                if f.lower().endswith('.png')
            ]
        except Exception as e:
            print(f"{folder}: PNG 파일 목록 읽기 실패 - {e}")
            continue

        for png_file in png_files:
            json_file = png_file.replace('.png', '.json').replace('.PNG', '.json')
            json_path = os.path.join(label_dir, json_file)

            if not os.path.exists(json_path):
                json_files = [
                    f for f in os.listdir(label_dir)
                    if f.lower() == json_file.lower()
                ]
                if not json_files:
                    missing_json_count += 1
                    print(f"[JSON 없음] {folder} / {png_file}")
                    continue
                json_path = os.path.join(label_dir, json_files[0])


            disease_name = CONFIG.disease_name_mapping.get(folder)
            if disease_name is None:
                missing_folders.add(folder)
                continue

            data_list.append({
                "img_path": os.path.normpath(os.path.join(img_dir, png_file)),
                "json_path": os.path.normpath(json_path),
                "label": disease_name,
                "risk": CONFIG.risk_mapping[disease_name],
                "class_idx": CONFIG.class_to_idx[disease_name]
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

train_csv_path = os.path.join(CONFIG.project_root, "train_index.csv")
valid_csv_path = os.path.join(CONFIG.project_root, "valid_index.csv")

# 1. 파일이 이미 존재하는지 확인
if os.path.exists(train_csv_path) and os.path.exists(valid_csv_path):
    print("=" * 70)
    print("✅ 기존 CSV 파일이 발견되어 로드합니다. (데이터 수집 건너뜀)")
    print("=" * 70)
    train_df = pd.read_csv(train_csv_path, encoding='utf-8-sig')
    valid_df = pd.read_csv(valid_csv_path, encoding='utf-8-sig')

else:
    # 2. 파일이 없으면 새로 생성 (최초 1회 실행)
    print("=" * 70)
    print("⚠️ CSV 파일이 없습니다. 데이터를 새로 수집합니다...")
    print("=" * 70)

    train_data = []
    valid_data = []

    # ---------------- Training ----------------
    print("\n[Dataset 14 - Training]")
    train_data.extend(collect_data_paths(CONFIG.dataset14_train_img, CONFIG.dataset14_train_label, "Dataset 14 Training"))
    print("\n[Dataset 15 - Training]")
    train_data.extend(collect_data_paths(CONFIG.dataset15_train_img, CONFIG.dataset15_train_label, "Dataset 15 Training"))

    # ---------------- Validation ----------------
    print("\n[Dataset 14 - Validation]")
    valid_data.extend(collect_data_paths(CONFIG.dataset14_val_img, CONFIG.dataset14_val_label, "Dataset 14 Validation"))
    print("\n[Dataset 15 - Validation]")
    valid_data.extend(collect_data_paths(CONFIG.dataset15_val_img, CONFIG.dataset15_val_label, "Dataset 15 Validation"))

    print(f"\nTraining 샘플 수   : {len(train_data):,}")
    print(f"Validation 샘플 수 : {len(valid_data):,}")

    if len(train_data) == 0:
        raise RuntimeError("Training 데이터가 0개입니다. 경로를 확인하세요.")

    # DataFrame 생성 및 저장
    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    
    train_df.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
    valid_df.to_csv(valid_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV 저장 완료:\n - {train_csv_path}\n - {valid_csv_path}")


# ============================================================
# Validation → Validation / Test split 

VAL_SPLIT_PATH = os.path.join(CONFIG.project_root, "val_index.csv")
TEST_SPLIT_PATH = os.path.join(CONFIG.project_root, "test_index.csv")

from sklearn.model_selection import train_test_split

if not os.path.exists(TEST_SPLIT_PATH):
    # 최초 1회만 split 수행
    val_df, test_df = train_test_split(
        valid_df,
        test_size=0.5,
        random_state=CONFIG.seed,
        stratify=valid_df["class_idx"]
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
        disease_label = torch.tensor(row["class_idx"], dtype=torch.long)
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

        disease_label = torch.tensor(row["class_idx"], dtype=torch.long)
        risk_label = torch.tensor(row["risk"], dtype=torch.long)

        return img, disease_label, risk_label

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

        # ===============================
        # 1. EfficientNet Backbone
        # ===============================
        self.backbone = timm.create_model(
            CONFIG.model_name,
            pretrained=CONFIG.is_pretrained,
            num_classes=0,
            global_pool="avg"
        )
        eff_dim = self.backbone.num_features

        # ===============================
        # 2. ResNet Backbone  (code.ref1 활용)
        # ===============================
        self.backbone_resnet = timm.create_model(
            CONFIG.model_name_resnet,
            pretrained=CONFIG.is_pretrained,
            num_classes=0,
            global_pool="avg"
        )
        res_dim = self.backbone_resnet.num_features

        # ===============================
        # 3. Feature Fusion
        # ===============================
        fusion_dim = eff_dim + res_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )

        # ===============================
        # 4. Heads (기존 구조 유지)
        # ===============================
        self.disease_head = nn.Sequential(
            nn.Linear(512, CONFIG.n_classes)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(512, 3)
        )

    # ===============================
    # 5. Forward (code.ref2 기반)
    # ===============================
    def forward(self, x):
        feats_eff = self.backbone(x)          # (B, eff_dim)
        feats_res = self.backbone_resnet(x)   # (B, res_dim)

        feats = torch.cat([feats_eff, feats_res], dim=1)
        fused = self.fusion(feats)

        disease_logits = self.disease_head(fused)
        risk_logits = self.risk_head(fused)

        return disease_logits, risk_logits

# ==========================
# Backbone Freeze Utilities
# ==========================

def freeze_resnet(model):
    for p in model.backbone_resnet.parameters():
        p.requires_grad = False


def freeze_efficientnet_except_last(model):
    for name, param in model.backbone.named_parameters():
        if "blocks.5" not in name and "blocks.6" not in name:
            param.requires_grad = False


           

# Hybrid weighting: inverse frequency × risk factor
class_counts = train_df["class_idx"].value_counts().sort_index()
inv_freq = 1.0 / class_counts

risk_factors = []
for idx in range(CONFIG.n_classes):
    cls_name = CONFIG.idx_to_class[idx]
    risk = CONFIG.risk_mapping[cls_name]
    if risk == 2:   # High risk (malignant)
        risk_factors.append(3.0)
    elif risk == 1: # Intermediate risk
        risk_factors.append(2.0)
    else:           # Low risk / benign
        risk_factors.append(1.0)

# Combine inverse frequency and risk multipliers
weights = inv_freq * risk_factors

# Normalize so average weight ≈ 1
weights = weights / weights.mean()

print("weights:", weights.tolist())

class_weights = torch.tensor(
    weights.to_numpy(),
    dtype=torch.float,
    device=CONFIG.device
)

criterion_disease = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)


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

    for images, disease_labels, risk_labels in bar:
        bs = images.size(0)

        images = images.to(CONFIG.device, dtype=torch.float)
        disease_labels = disease_labels.to(CONFIG.device)
        risk_labels = risk_labels.to(CONFIG.device)

        optimizer.zero_grad()

        disease_logits, risk_logits = model(images)

        loss_d = criterion_disease(disease_logits, disease_labels)
        loss_r = criterion_risk(risk_logits, risk_labels)

        loss = loss_d + RISK_LOSS_WEIGHT * loss_r
        loss.backward()
        optimizer.step()

        preds = disease_logits.argmax(dim=1)

        y_preds.append(preds.detach().cpu().numpy())
        y_trues.append(disease_labels.detach().cpu().numpy())

        running_loss += loss.item() * bs
        dataset_size += bs

        epoch_loss = running_loss / dataset_size
        epoch_acc = (np.concatenate(y_preds) == np.concatenate(y_trues)).mean()

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

    for images, disease_labels, risk_labels in bar:
        bs = images.size(0)

        images = images.to(CONFIG.device, dtype=torch.float)
        disease_labels = disease_labels.to(CONFIG.device)
        risk_labels = risk_labels.to(CONFIG.device)

        disease_logits, risk_logits = model(images)

        loss_d = criterion_disease(disease_logits, disease_labels)
        loss_r = criterion_risk(risk_logits, risk_labels)

        loss = loss_d + RISK_LOSS_WEIGHT * loss_r

        preds = disease_logits.argmax(dim=1)

        y_preds.append(preds.cpu().numpy())
        y_trues.append(disease_labels.cpu().numpy())

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
    """
    21-Class 분류 모델의 종합 평가
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    all_probs = []
    all_risk_preds = []
    all_risk_trues = []
    
    print(f"\n{'='*70}")
    print(f"📊 {dataset_name} Set Evaluation")
    print(f"{'='*70}\n")
    
    # 예측 수집
    for images, disease_labels, risk_labels in tqdm(
        data_loader, desc=f"{dataset_name} Evaluation"):

        images = images.to(CONFIG.device, dtype=torch.float)
        disease_labels = disease_labels.to(CONFIG.device, dtype=torch.long)
        risk_labels = risk_labels.to(CONFIG.device, dtype=torch.long)

        disease_logits, risk_logits = model(images)

        disease_probs = torch.softmax(disease_logits, dim=1)
        disease_preds = disease_logits.argmax(dim=1)
        risk_preds = risk_logits.argmax(dim=1)

        all_preds.append(disease_preds.cpu().numpy())
        all_trues.append(disease_labels.cpu().numpy())
        all_probs.append(disease_probs.cpu().numpy())
        all_risk_preds.append(risk_preds.cpu().numpy())
        all_risk_trues.append(risk_labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    y_prob = np.concatenate(all_probs)
    y_pred_risk = np.concatenate(all_risk_preds)
    y_true_risk = np.concatenate(all_risk_trues)
    
    # 2. 기본 메트릭 계산
    
    # Overall Accuracy
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (per-class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        zero_division=0
    )
    
    # Weighted F1 (클래스 크기 고려)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='weighted',
        zero_division=0
    )
    
    # Macro F1 (모든 클래스 동등)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. 결과 출력 - 기본 메트릭
    print(f"📈 Overall Performance")
    print(f"{'─'*70}")
    print(f"  Overall Accuracy    : {overall_acc:.4f}")
    print(f"  Weighted Precision  : {weighted_precision:.4f}")
    print(f"  Weighted Recall     : {weighted_recall:.4f}")
    print(f"  Weighted F1-Score   : {weighted_f1:.4f}")
    print(f"  Macro Precision     : {macro_precision:.4f}")
    print(f"  Macro Recall        : {macro_recall:.4f}")
    print(f"  Macro F1-Score      : {macro_f1:.4f}")
    
    # 4. Per-Class 성능 (표 형식)
    print(f"\n📋 Per-Class Performance")
    print(f"{'─'*70}")
    print(f"{'Class':<20} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8} {'Risk':<4}")
    print(f"{'─'*70}")
    
    for idx, class_name in enumerate(CONFIG.class_names):
        risk = CONFIG.risk_mapping[class_name]
        risk_label = ['Low', 'Int', 'High'][risk]
        
        print(
            f"{class_name:<20} "
            f"{precision[idx]:>6.3f} "
            f"{recall[idx]:>6.3f} "
            f"{f1[idx]:>6.3f} "
            f"{support[idx]:>8,d} "
            f"{risk_label:<4}"
        )
    
    # 5. High Risk 클래스 집중 분석
    print(f"\n🚨 High Risk Classes (Malignant) - 최우선 지표")
    print(f"{'─'*70}")
    
    high_risk_classes = ['악성흑색종', '기저세포암', '편평세포암']
    high_risk_indices = [CONFIG.class_to_idx[name] for name in high_risk_classes]
    
    high_risk_recalls = []
    high_risk_precisions = []
    
    for class_name in high_risk_classes:
        idx = CONFIG.class_to_idx[class_name]
        
        # Sensitivity (Recall)
        mask_true = (y_true == idx)
        if mask_true.sum() > 0:
            class_recall = (y_pred[mask_true] == idx).mean()
        else:
            class_recall = 0.0
        
        # Precision
        mask_pred = (y_pred == idx)
        if mask_pred.sum() > 0:
            class_precision = (y_true[mask_pred] == idx).mean()
        else:
            class_precision = 0.0
        
        high_risk_recalls.append(class_recall)
        high_risk_precisions.append(class_precision)
        
        print(
            f"  {class_name:<15}: "
            f"Recall={class_recall:.4f} | "
            f"Precision={class_precision:.4f} | "
            f"Samples={mask_true.sum():,d}"
        )
    
    # High Risk 평균
    avg_high_risk_recall = np.mean(high_risk_recalls)
    avg_high_risk_precision = np.mean(high_risk_precisions)
    
    print(f"\n  {'⭐ Average High Risk':<15}: "
          f"Recall={avg_high_risk_recall:.4f} | "
          f"Precision={avg_high_risk_precision:.4f}")
    
    # 6. Risk-Level Performance (위험도 단위 평가)
    print(f"\n🎯 Risk-Level Performance")
    print(f"{'─'*70}")
    
    # 실제 위험도 vs 예측 위험도
    print(f"\n🎯 Risk-Level Performance")
    print(f"{'─'*70}")
    
    risk_sensitivities = {}
    risk_fnrs = {}
    
    for risk_level in [2, 1, 0]:
        risk_name = ['Low', 'Intermediate', 'High'][risk_level]
        mask = (y_true_risk == risk_level)
        
        if mask.sum() == 0:
            level_sens = float("nan")
            level_fnr = float("nan")
        else:
            tp = (y_pred_risk[mask] == risk_level).sum()
            fn = mask.sum() - tp
            
            level_sens = tp / (tp + fn)
            level_fnr = fn / (tp + fn)
        
        risk_sensitivities[risk_name] = level_sens
        risk_fnrs[risk_name] = level_fnr
        
        print(f"  {risk_name:<18}: Sens={level_sens:.4f} | FNR={level_fnr:.4f}")
    
    # 전체 위험도 Accuracy
    risk_acc = (y_true_risk == y_pred_risk).mean()
    print(f"\n  Overall Risk Accuracy: {risk_acc:.4f}")
    
    # 7. 의료적으로 의미 있는 AUC
    print(f"\n📊 Clinically Relevant AUC")
    print(f"{'─'*70}")
    
    # 7.1 High Risk (Malignant) vs Others
    y_high_risk_binary = np.isin(y_true, high_risk_indices).astype(int)
    y_high_risk_prob = y_prob[:, high_risk_indices].sum(axis=1)
    
    try:
        high_risk_auc = roc_auc_score(y_high_risk_binary, y_high_risk_prob)
        print(f"  High Risk vs Others AUC    : {high_risk_auc:.4f}")
    except:
        print(f"  High Risk vs Others AUC    : N/A")
    
    # ============================================================
    # 7.2 Risk-Level One-vs-Rest AUC (FIXED)
    # ============================================================

    # risk level → class indices 매핑
    risk_to_class_indices = {
        2: [CONFIG.class_to_idx[name] for name, r in CONFIG.risk_mapping.items() if r == 2],
        1: [CONFIG.class_to_idx[name] for name, r in CONFIG.risk_mapping.items() if r == 1],
    }

    # High Risk (2) vs Others
    y_risk_high = (y_true_risk == 2).astype(int)
    y_risk_high_prob = y_prob[:, risk_to_class_indices[2]].sum(axis=1)

    if y_risk_high.sum() > 0 and (1 - y_risk_high).sum() > 0:
        risk_high_auc = roc_auc_score(y_risk_high, y_risk_high_prob)
        print(f"  Risk Level 2 (High) AUC    : {risk_high_auc:.4f}")
    # Intermediate Risk (1) vs Others
    y_risk_int = (y_true_risk == 1).astype(int)
    y_risk_int_prob = y_prob[:, risk_to_class_indices[1]].sum(axis=1)

    if y_risk_int.sum() > 0 and (1 - y_risk_int).sum() > 0:
        risk_int_auc = roc_auc_score(y_risk_int, y_risk_int_prob)
        print(f"  Risk Level 1 (Int) AUC     : {risk_int_auc:.4f}")

    
    # 7.3 Multi-class One-vs-Rest (참고용)
    present_labels = np.unique(y_true)

    try:
        ovr_auc = roc_auc_score(
            y_true,
            y_prob[:, present_labels],
            labels=present_labels,
            multi_class='ovr',
            average='macro'
        )
        print(f"  21-Class OvR AUC (참고)     : {ovr_auc:.4f}")
    except ValueError:
        print(f"  21-Class OvR AUC (참고)     : N/A")
    
    # 8. Confusion Matrix 저장
    print(f"\n💾 Saving Confusion Matrix...")
    
    # CSV 저장
    cm_df = pd.DataFrame(
        cm,
        index=CONFIG.class_names,
        columns=CONFIG.class_names
    )
    cm_path = CONFIG.project_root / f"{dataset_name.lower()}_confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding='utf-8-sig')
    print(f"  ✅ CSV saved: {cm_path}")
    
    # 시각화 저장
    try:
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=CONFIG.class_names,
            yticklabels=CONFIG.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.title(f'{dataset_name} Set Confusion Matrix', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_img_path = CONFIG.project_root / f"{dataset_name.lower()}_confusion_matrix.png"
        plt.savefig(cm_img_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Image saved: {cm_img_path}")
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Visualization failed: {e}")
    
    # 9. Classification Report (상세)
    print(f"\n📄 Detailed Classification Report")
    print(f"{'─'*70}")
    # ============================================================
    # 9. Classification Report (Safe for missing classes)
    
    present_labels = np.unique(y_true)
    present_class_names = [CONFIG.class_names[i] for i in present_labels]
    print(classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        zero_division=0
    ))

    
    # 10. 결과 딕셔너리 반환
    results = {
        'overall_accuracy': overall_acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'high_risk_recall': avg_high_risk_recall,
        'high_risk_precision': avg_high_risk_precision,
        'risk_accuracy': risk_acc,
        'risk_sensitivities': risk_sensitivities,  # ✅ 추가
        'risk_fnrs': risk_fnrs,  # ✅ 추가
        'confusion_matrix': cm,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
    }
    
    print(f"\n{'='*70}")
    print(f"✅ Evaluation Complete")
    print(f"{'='*70}\n")
    
    return results

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
    
    # ========================================
    # 1. 학습 곡선 (Loss & Accuracy)
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Performance Dashboard', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1-1. Loss 곡선
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['valid_loss'], 'r-s', label='Valid Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
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
    # 2. Per-Class 성능 비교 (평가 결과 있을 때)
    # ========================================
    if results is not None:
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
        
        class_names = CONFIG.class_names
        precision = results['per_class_precision']
        recall = results['per_class_recall']
        f1 = results['per_class_f1']
        support = results['per_class_support']
        
        # 위험도별 색상
        colors = []
        for name in class_names:
            risk = CONFIG.risk_mapping[name]
            if risk == 2:
                colors.append('red')
            elif risk == 1:
                colors.append('orange')
            else:
                colors.append('green')
        
        x = np.arange(len(class_names))
        
        # 2-1. Precision 비교
        axes[0, 0].bar(x, precision, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        axes[0, 0].set_ylabel('Precision', fontsize=12)
        axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axhline(y=precision.mean(), color='blue', linestyle='--', 
                          label=f'Avg: {precision.mean():.3f}', linewidth=2)
        axes[0, 0].legend()
        
        # 2-2. Recall (Sensitivity) 비교
        axes[0, 1].bar(x, recall, color=colors, alpha=0.7)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Recall (Sensitivity)', fontsize=12)
        axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].axhline(y=recall.mean(), color='blue', linestyle='--', 
                          label=f'Avg: {recall.mean():.3f}', linewidth=2)
        axes[0, 1].legend()
        
        # 2-3. F1-Score 비교
        axes[1, 0].bar(x, f1, color=colors, alpha=0.7)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('F1-Score', fontsize=12)
        axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].axhline(y=f1.mean(), color='blue', linestyle='--', 
                          label=f'Avg: {f1.mean():.3f}', linewidth=2)
        axes[1, 0].legend()
        
        # 2-4. Support (샘플 수) 비교
        axes[1, 1].bar(x, support, color=colors, alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
        axes[1, 1].set_title('Support (Sample Count) by Class', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='High Risk'),
            Patch(facecolor='orange', alpha=0.7, label='Intermediate Risk'),
            Patch(facecolor='green', alpha=0.7, label='Low Risk')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        save_path = save_dir / 'per_class_performance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class performance saved: {save_path}")
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
            axes[0].set_ylabel('Sensitivity (Recall)', fontsize=12)
            axes[0].set_title('Sensitivity by Risk Level', fontsize=14, fontweight='bold')
            axes[0].set_ylim([0, 1])
            axes[0].grid(axis='y', alpha=0.3)
            
            # 값 표시
            for bar, val in zip(bars, sensitivities):
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # 3-2. False Negative Rate
            bars = axes[1].bar(risk_names, fnrs, color=risk_colors, alpha=0.7, edgecolor='black')
            axes[1].set_ylabel('False Negative Rate', fontsize=12)
            axes[1].set_title('FNR by Risk Level (Lower is Better)', fontsize=14, fontweight='bold')
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
        
        # ========================================
        # 4. 주요 지표 요약 카드
        # ========================================
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 제목
        fig.text(0.5, 0.95, 'Model Performance Summary', 
                ha='center', fontsize=20, fontweight='bold')
        
        # 메트릭 박스
        metrics_text = f"""
        
        📊 Overall Metrics
        {'─'*50}
        Overall Accuracy      : {results['overall_accuracy']:.4f}
        Weighted F1-Score     : {results['weighted_f1']:.4f}
        Macro F1-Score        : {results['macro_f1']:.4f}
        
        
        🚨 High Risk Performance (Most Important)
        {'─'*50}
        High Risk Recall      : {results['high_risk_recall']:.4f}
        High Risk Precision   : {results['high_risk_precision']:.4f}
        
        
        🎯 Risk-Level Accuracy
        {'─'*50}
        Overall Risk Accuracy : {results['risk_accuracy']:.4f}
        
        
        📈 Per-Risk Sensitivity
        {'─'*50}
        High (악성)           : {results['risk_sensitivities']['High']:.4f}
        Intermediate (중등도)  : {results['risk_sensitivities']['Intermediate']:.4f}
        Low (양성)            : {results['risk_sensitivities']['Low']:.4f}
        """
        
        fig.text(0.1, 0.5, metrics_text, 
                fontsize=12, verticalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        save_path = save_dir / 'performance_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance summary saved: {save_path}")
        plt.close()
    
    print(f"\n{'='*70}")
    print(f"✅ All visualizations saved to: {save_dir}")
    print(f"{'='*70}\n")



if __name__ == "__main__":
    # 1. 모델 초기화
    model = SkinDiseaseModel().to(CONFIG.device)
    
    freeze_resnet(model)
    freeze_efficientnet_except_last(model)


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
    RUN_TEST = False
    
    if RUN_TEST:
        print("\n🚀 Test set evaluation started")
        
        test_results = comprehensive_evaluation(
            model=model,
            data_loader=test_loader,
            dataset_name="Test"
        )
        
        # Test 결과 시각화
        print("\n🎨 Generating test visualizations...")
        visualize_model_performance(
            history=history,
            results=test_results,
            save_dir=CONFIG.project_root / "test_results"
        )
        
        print("\n" + "="*70)
        print("Key Performance Indicators")
        print("="*70)
        print(f"  Overall Accuracy     : {test_results['overall_accuracy']:.4f}")
        print(f"  Weighted F1-Score    : {test_results['weighted_f1']:.4f}")
        print(f"  High Risk Recall     : {test_results['high_risk_recall']:.4f} ⭐")
        print(f"  Risk-Level Accuracy  : {test_results['risk_accuracy']:.4f}")
        print("="*70)

    # 8. Training history 저장
    history_df = pd.DataFrame(history)
    history_path = CONFIG.project_root / "training_history2.csv"
    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
    print(f"📁 Training history saved: {history_path}")