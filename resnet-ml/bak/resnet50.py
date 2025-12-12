# ============================================
# 1) 기본 라이브러리 임포트
# ============================================
import numpy as np
import pandas as pd
import os
import pathlib
import time   # 전체 학습 시간 측정용
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Lightning 콜백: 학습률 모니터링, epoch 시간 측정
from pytorch_lightning.callbacks import LearningRateMonitor, Timer, RichProgressBar


# ============================================
# 2) MPS 디바이스 설정
# ============================================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("사용 디바이스:", device)


# ============================================
# 3) EuroSAT 데이터 경로 설정 (로컬 버전)
# ============================================
dataset_root = pathlib.Path("./data/EuroSAT/")

train_df = pd.read_csv(dataset_root / "train.csv").reset_index(drop=True)
valid_df = pd.read_csv(dataset_root / "validation.csv").reset_index(drop=True)
test_df  = pd.read_csv(dataset_root / "test.csv").reset_index(drop=True)


# ============================================
# 4) 커스텀 Dataset
# ============================================
class EuroSATDataset(Dataset):
    def __init__(self, annotation_df, transform=None):
        super().__init__()
        self.data = annotation_df
        self.transform = transform or v2.Compose([
            v2.ToDtype(torch.float32, scale=True),  
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        image_path = (dataset_root / sample["Filename"]).as_posix()
        img = read_image(image_path)
        label = torch.tensor(sample["Label"])

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================
# 5) Lightning DataModule
# ============================================
class EuroSATLightningDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_data = EuroSATDataset(train_df)
        self.valid_data = EuroSATDataset(valid_df)
        self.test_data  = EuroSATDataset(test_df)

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.valid_data, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_data, shuffle=False)


# ============================================
# 6) Lightning Module (모델 + 학습 루프)
# ============================================
class EuroSATLightningModule(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-4):
        super().__init__()
        self.lr = lr
        
        # ImageNet pretrained ResNet-50
        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validate_or_test(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y, task="multiclass", num_classes=10)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.validate_or_test(batch, "val")

    def test_step(self, batch, batch_idx):
        self.validate_or_test(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


# ============================================
# 7) 학습 실행 (시간 측정 포함)
# ============================================
start_time = time.time()  # 전체 학습 시간 측정 시작

datamodule = EuroSATLightningDataModule(batch_size=32)
model = EuroSATLightningModule(num_classes=10, lr=1e-4)

# epoch 별 시간 추적 타이머
timer = Timer(interval="epoch", verbose=True)

# 학습률 모니터링 콜백
lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    max_epochs=20,
    accelerator="mps" if torch.backends.mps.is_available() else "cpu",
    devices=1,
    log_every_n_steps=10,
    callbacks=[
        lr_monitor,   # 학습률 기록
        timer,        # epoch 시간 기록
        RichProgressBar(),  # 강력한 progress bar
    ]
)

print("📌 학습 시작!")
trainer.fit(model, datamodule)
trainer.test(model, datamodule)

end_time = time.time()
print(f"\n⏱ 전체 학습 소요 시간: { (end_time - start_time) / 60:.2f} 분")

print("\n⏱ Epoch 별 소요 시간:")
print(timer.time_elapsed())


# ============================================
# 8) 모델 저장
# ============================================
save_path = "./best_resnet50_eurosat_mps.pth"
torch.save(model.state_dict(), save_path)
print("\n💾 모델 저장 완료:", save_path)