import numpy as np
from .metrics import eval_metrics

class ZeroShotClassifier:
    def predict(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray, unique_labels, target_labels):
        # cosine similarity (embeddings are expected to be L2-normalized)
        scores = image_embeddings.dot(text_embeddings.T)
        preds = [unique_labels[np.argmax(row)] for row in scores]
        return preds

    def evaluate(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray, unique_labels, target_labels):
        preds = self.predict(image_embeddings, text_embeddings, unique_labels, target_labels)
        test_metrics = eval_metrics(target_labels, preds)
        train_metrics = eval_metrics(target_labels, preds)
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"
        return train_metrics, test_metrics
