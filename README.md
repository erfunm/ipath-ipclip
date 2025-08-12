# IP-CLIP (IPATH): Clean Reproducible Code

This repository contains a clean, minimal implementation used in our paper **IPATH** where we fine-tune a pre-trained CLIP model on our Instagram-derived pathology dataset to obtain **IP-CLIP**.

> **Abstract (short):** We built the IPATH dataset and fine-tuned CLIP to create IP-CLIP. Across two external histopathology datasets, IP-CLIP outperformed baseline CLIP on **zero-shot classification** (F1 **0.40–0.71** vs **0.19–0.31**) and **linear probing** (F1 **0.89–0.91** vs **0.73–0.85**), highlighting the value of social media data for medical/veterinary image classification.

## What’s here

- `ipclip/embedders.py` – Simple wrapper around OpenAI CLIP for encoding images and text.
- `ipclip/datasets.py` – Minimal PyTorch datasets used by training/eval.
- `ipclip/transforms.py` – Train/eval transforms.
- `ipclip/scheduler.py` – Warmup + cosine LR schedule.
- `ipclip/finetune.py` – Fine-tuning a linear head on top of CLIP image embeddings.
- `ipclip/linear_probe.py` – Linear probing on frozen embeddings.
- `ipclip/zero_shot.py` – Zero-shot prediction/evaluation.
- `ipclip/retrieval.py` – Simple text→image retrieval utilities.
- `ipclip/metrics.py` – Evaluation metrics.
- `ipclip/caching.py` – (Optional) tiny on-disk caching helper.
- `ipclip/utils.py` – Append results to CSV.
- `scripts/*` – CLI scripts for fine-tuning and evaluation.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # (linux/mac)
# or: py -m venv .venv && .venv\Scripts\activate  # (windows)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA/CPU build
pip install numpy pandas scikit-learn pillow tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Quickstart

### Zero-shot evaluation
```bash
python scripts/eval_zero_shot.py \  --images_csv /path/to/Kather_test.csv \  --labels "adipose tissue" "background" "debris" "lymphocytes" "mucus" "smooth muscle" "normal colon mucosa" "cancer-associated stroma" "colorectal adenocarcinoma epithelium"
```

### Linear probing
```bash
python scripts/eval_linear_probe.py \  --train_csv /path/to/Kather_train.csv \  --test_csv /path/to/Kather_test.csv \  --alpha 0.001
```

### Fine-tuning (IP-CLIP linear head)
```bash
python scripts/finetune_ipclip.py \  --num_classes 9 \  --train_csv /path/to/Kather_train.csv \  --val_csv /path/to/Kather_test.csv \  --epochs 5 --batch_size 8 --lr 5e-5
```

## Datasets

CSV format expected by scripts:
- **Zero-shot**: `images_csv` must have columns: `image`, `label` (where `label` is the ground-truth class string)
- **Linear probing / Fine-tuning**: `*_csv` must have columns: `image`, `label` (numeric or string)

You can adapt your existing preprocessing code to export this CSV format.

## Notes

- Code is intentionally minimal and readable. All legacy/unused files from the original repo have been removed.
- We rely on **OpenAI CLIP** (`clip` Python package) for encoding and on standard PyTorch/Sklearn for training/evaluation.
- Feel free to drop this folder into a new GitHub repo as-is.
