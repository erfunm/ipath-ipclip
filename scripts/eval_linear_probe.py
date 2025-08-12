import argparse, pandas as pd, numpy as np
from ipclip.embedders import build_embedder
from ipclip.linear_probe import LinearProber

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B/32")
    ap.add_argument("--train_csv", required=True, help="CSV with columns [image,label]")
    ap.add_argument("--test_csv", required=True, help="CSV with columns [image,label]")
    ap.add_argument("--alpha", type=float, default=1e-3)
    return ap.parse_args()

def main():
    args = parse_args()
    df_tr = pd.read_csv(args.train_csv)
    df_te = pd.read_csv(args.test_csv)
    e = build_embedder(arch=args.arch)
    Xtr = e.embed_images(df_tr["image"].tolist())
    Xte = e.embed_images(df_te["image"].tolist())
    ytr = df_tr["label"].tolist()
    yte = df_te["label"].tolist()
    prober = LinearProber(alpha=args.alpha)
    clf, (test_metrics, train_metrics) = prober.train_and_test(Xtr, ytr, Xte, yte)
    print("Linear probe (test):", test_metrics)

if __name__ == "__main__":
    main()
