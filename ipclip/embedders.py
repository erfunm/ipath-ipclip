from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

from .datasets import CLIPImageDataset, CLIPCaptioningDataset

# We use OpenAI CLIP (pip install git+https://github.com/openai/CLIP.git)
import clip


class CLIPEmbedder:
    """Generic wrapper around a CLIP model for encoding images and text."""

    def __init__(self, model, preprocess, name: str, backbone_path: str | None = None):
        self.model = model
        self.preprocess = preprocess
        self.name = name
        self.backbone_path = backbone_path

    @torch.no_grad()
    def embed_images(self, image_paths: List[str], device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     batch_size: int = 32, num_workers: int = 1) -> np.ndarray:
        ds = CLIPImageDataset(image_paths, self.preprocess)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        embs = []
        pbar = tqdm.tqdm(total=max(1, len(image_paths) // max(1, batch_size)), position=0, desc="Encoding images")
        self.model.eval().to(device)
        for imgs in dl:
            imgs = imgs.to(device)
            feats = self.model.encode_image(imgs).detach().cpu().numpy()
            embs.extend(feats)
            pbar.update(1)
        pbar.close()
        embs = np.asarray(embs, dtype=np.float32)
        # L2-normalize
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return embs

    @torch.no_grad()
    def embed_text(self, captions: List[str], device: str = "cuda" if torch.cuda.is_available() else "cpu",
                   batch_size: int = 64, num_workers: int = 1) -> np.ndarray:
        ds = CLIPCaptioningDataset(captions)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        embs = []
        pbar = tqdm.tqdm(total=max(1, len(captions) // max(1, batch_size)), position=0, desc="Encoding text")
        self.model.eval().to(device)
        for batch_caps in dl:
            tokens = clip.tokenize(batch_caps, truncate=True).to(device)
            feats = self.model.encode_text(tokens).detach().cpu().numpy()
            embs.extend(feats)
            pbar.update(1)
        pbar.close()
        embs = np.asarray(embs, dtype=np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        return embs


def load_clip(arch: str = "ViT-B/32", device: str | None = None) -> Tuple[torch.nn.Module, any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(arch, device=device, jit=False)
    model.eval()
    return model, preprocess


def build_embedder(model_name: str = "clip", arch: str = "ViT-B/32") -> CLIPEmbedder:
    model, preprocess = load_clip(arch=arch)
    return CLIPEmbedder(model, preprocess, name=model_name, backbone_path=None)
