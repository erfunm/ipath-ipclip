import numpy as np
from .metrics import retrieval_metrics

class ImageRetrieval:
    def topk(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray, k: int = 50):
        # Return indices of the top-k images for each text embedding
        sims = text_embeddings.dot(image_embeddings.T)  # [T, I]
        return np.argsort(sims, axis=1)[:, -k:][:, ::-1]

    def evaluate(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray):
        nn = self.topk(image_embeddings, text_embeddings, k=50)
        targets = list(range(0, len(image_embeddings)))
        test_metrics = retrieval_metrics(targets, nn)
        train_metrics = retrieval_metrics(targets, nn)
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"
        return train_metrics, test_metrics
