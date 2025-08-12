from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
import clip

from .datasets import CLIPImageLabelDataset
from .scheduler import cosine_lr


class LinearClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        return self.fc(x)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


@dataclass
class FTArgs:
    model_name: str = "clip"  # "clip" or "plip" (treated the same here)
    arch: str = "ViT-B/32"
    optimizer: str = "AdamW"
    pxsize: int = 224
    first_resize: int = 512


class FineTuner:
    def __init__(self, args: Optional[FTArgs] = None, num_classes: int = 2, lr: float = 5e-5,
                 weight_decay: float = 0.2, warmup: int = 0):
        self.args = args or FTArgs()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP backbone
        self.model, self.preprocess = clip.load(self.args.arch, device=self.device, jit=False)

        # 512 is the embedding dim for ViT-B/32
        self.linear = LinearClassifier(512, num_classes).to(self.device)

        # Optimizer
        params = list(self.model.parameters()) + list(self.linear.parameters())
        if self.args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        self.warmup = warmup

        if self.device != "cpu":
            clip.model.convert_weights(self.model)

    @torch.no_grad()
    def _forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._forward_features(images)
        return self.linear(feats)

    def _validate(self, dl: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        self.linear.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []
        for imgs, labels in dl:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                logits = self._forward(imgs)
                loss = self.criterion(logits, labels)
            total_loss += float(loss.detach().cpu())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        preds = logits.argmax(dim=1).numpy()
        labels_np = labels.numpy()

        # Macro and weighted F1
        from sklearn.metrics import f1_score
        f1_weighted = f1_score(labels_np, preds, average="weighted")
        f1_macro = f1_score(labels_np, preds, average="macro")

        self.model.train()
        self.linear.train()
        return total_loss, f1_weighted, f1_macro

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, batch_size: int = 8, epochs: int = 5,
            evaluation_steps: int = 500, num_workers: int = 2) -> pd.DataFrame:
        train_ds = CLIPImageLabelDataset(df_train, self.preprocess)
        val_ds = CLIPImageLabelDataset(df_val, self.preprocess)
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

        num_batches = len(train_dl)
        total_steps = max(1, num_batches * epochs)
        scheduler = cosine_lr(self.optimizer, base_lr=self.optimizer.param_groups[0]["lr"], warmup_length=self.warmup, steps=total_steps)

        hist = []
        for epoch in range(epochs):
            pbar = tqdm.tqdm(total=num_batches, desc=f"epoch {epoch+1}/{epochs}")
            for i, (imgs, labels) in enumerate(train_dl):
                step = epoch * num_batches + i
                self.optimizer.zero_grad(set_to_none=True)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self._forward(imgs)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.device != "cpu":
                    convert_models_to_fp32(self.model)
                self.optimizer.step()
                if self.device != "cpu":
                    clip.model.convert_weights(self.model)
                scheduler(step)
                pbar.update(1)

                if evaluation_steps and (step % evaluation_steps == 0):
                    val_loss, f1_w, f1_m = self._validate(val_dl)
            pbar.close()
            val_loss, f1_w, f1_m = self._validate(val_dl)
            hist.append({"epoch": epoch+1, "val_loss": val_loss, "f1_weighted": f1_w, "f1_macro": f1_m})
        return pd.DataFrame(hist)
