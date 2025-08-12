import argparse, pandas as pd
from ipclip.embedders import build_embedder
from ipclip.zero_shot import ZeroShotClassifier

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ViT-B/32")
    ap.add_argument("--images_csv", required=True, help="CSV with columns [image,label]")
    ap.add_argument("--labels", nargs="+", required=True, help="List of text labels in the same order used for evaluation")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.images_csv)
    e = build_embedder(arch=args.arch)
    img_vecs = e.embed_images(df["image"].tolist())
    txt_vecs = e.embed_text(args.labels)
    clf = ZeroShotClassifier()
    train_metrics, test_metrics = clf.evaluate(img_vecs, txt_vecs, unique_labels=args.labels, target_labels=df["label"].tolist())
    print("Zero-shot (test):", test_metrics)

if __name__ == "__main__":
    main()
