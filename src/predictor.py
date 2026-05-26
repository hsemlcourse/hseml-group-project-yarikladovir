from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "final_tfidf_linear_svc.joblib"
MODEL_NAME = "TF-IDF + Linear SVC"


_model: Optional[Any] = None


def load_model() -> Any:
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

    return _model


def _to_bool_fraud(prediction: Any) -> Optional[bool]:
    value = str(prediction).strip().lower()

    if value in {"1", "true", "fraud", "fraudulent", "fake", "scam"}:
        return True

    if value in {"0", "false", "real", "legit", "not_fraud", "non-fraudulent"}:
        return False

    return None


def _get_score(model: Any, text: str) -> Optional[float]:
    try:
        if hasattr(model, "decision_function"):
            raw_score = model.decision_function([text])
            return float(np.ravel(raw_score)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])
            return float(np.max(proba))

    except Exception:
        return None

    return None


def predict_text(text: str) -> dict:
    model = load_model()

    prediction = model.predict([text])[0]
    score = _get_score(model, text)

    if isinstance(prediction, np.generic):
        prediction = prediction.item()

    return {
        "prediction": prediction,
        "is_fraud": _to_bool_fraud(prediction),
        "score": score,
        "model_name": MODEL_NAME,
    }
