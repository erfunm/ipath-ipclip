import argparse, os, pandas as pd
from ipclip.finetune import FineTuner, FTArgs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B/32")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    ft = FineTuner(FTArgs(arch=args.arch), num_classes=args.num_classes, lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup)
    hist = ft.fit(df_train, df_val, batch_size=args.batch_size, epochs=args.epochs, evaluation_steps=200)
    out = os.path.splitext(os.path.basename(args.train_csv))[0] + "_finetune_history.csv"
    hist.to_csv(out, index=False)
    print("Saved:", out)

if __name__ == "__main__":
    main()
