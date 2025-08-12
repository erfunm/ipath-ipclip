import numpy as np
from typing import List
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from .metrics import eval_metrics

class LinearProber:
    def __init__(self, alpha: float = 1e-3, seed: int = 7):
        self.alpha = alpha
        self.seed = seed
        self.clf = SGDClassifier(random_state=self.seed, loss="log_loss",
                                 alpha=self.alpha, penalty="l2",
                                 max_iter=10000, class_weight="balanced")

    def train_and_test(self, train_x: List, train_y: List, test_x: List, test_y: List):
        le = LabelEncoder()
        y_train = le.fit_transform(train_y)
        y_test = le.transform(test_y)
        self.clf.fit(train_x, y_train)
        pred_test = self.clf.predict(test_x)
        pred_train = self.clf.predict(train_x)
        test_metrics = eval_metrics(y_test, pred_test, average_method="macro")
        train_metrics = eval_metrics(y_train, pred_train, average_method="macro")
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"
        return self.clf, (test_metrics, train_metrics)
