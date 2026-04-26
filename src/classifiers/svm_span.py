from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SPAN_CLASSES: Tuple[str, ...] = ("ClinicalImpacts", "SocialImpacts")

@dataclass
class SVMTrainReport:
    best_params: Dict
    cv_macro_f1: float
    n_train: int
    class_counts: Dict[str, int]

def _build_pipeline(n_pca: int) -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=42)),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ])

def fit_svm(
    X: np.ndarray,
    y: np.ndarray,
    n_pca: int = 128,
    cv_folds: int = 5,
    C_grid: Optional[Sequence[float]] = None,
    gamma_grid: Optional[Sequence] = None,
    n_jobs: int = -1,
) -> Tuple[Pipeline, SVMTrainReport]:
    n_pca_eff = min(n_pca, X.shape[1], max(1, X.shape[0] - 1))
    pipeline = _build_pipeline(n_pca=n_pca_eff)

    param_grid = {
        "svc__C": list(C_grid or [0.1, 1.0, 10.0]),
        "svc__gamma": list(gamma_grid or ["scale", 0.01, 0.1]),
    }
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    gs.fit(X, y)

    labels, counts = np.unique(y, return_counts=True)
    report = SVMTrainReport(
        best_params={k.replace("svc__", ""): v for k, v in gs.best_params_.items()},
        cv_macro_f1=float(gs.best_score_),
        n_train=int(X.shape[0]),
        class_counts={str(lab): int(c) for lab, c in zip(labels, counts)},
    )
    return gs.best_estimator_, report

class SVMSpanClassifier:

    def __init__(self, pipeline: Pipeline, classes: Sequence[str] = SPAN_CLASSES) -> None:
        self.pipeline = pipeline
        self.classes = tuple(classes)

    def classify_batch(self, embeddings: np.ndarray) -> List[str]:
        if embeddings.size == 0:
            return []
        preds = self.pipeline.predict(embeddings)
        return [str(p) for p in preds]

    def save(self, path: str | Path, meta: Optional[Dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "classes": list(self.classes),
                "meta": meta or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SVMSpanClassifier":
        blob = joblib.load(path)
        return cls(pipeline=blob["pipeline"], classes=blob.get("classes", SPAN_CLASSES))
