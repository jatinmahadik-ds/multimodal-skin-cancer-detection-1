"""
model_loader.py
================================================================================
Loads the verified pre-trained models + schema ONCE at application startup:

    Model 1: EfficientNetB0 (ISIC 2019)               -> dermoscopic image cancer probability
    Model 2: Clinical-only XGBoost (PAD-UFES-20)       -> clinical cancer score (53 features)

Note on ResNet50:
    ResNet50 has been disconnected from the active inference pipeline.
    It is not loaded into memory and is not executed during prediction.

Note on Ensemble:
    No two-model ensemble combination has been statistically validated.
    EfficientNetB0 and Clinical XGBoost are operated as independent,
    uncombined models. No artificial combined weights are applied.
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# Force CPU-only for inference and ensure writable keras cache
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".keras_cache"))

# Base paths (repository-relative by default, overridable for external artifacts)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(
    os.environ.get("SKIN_MODELS_DIR", str(REPO_ROOT / "training" / "models"))
).expanduser().resolve()
XGBOOST_DIR = Path(
    os.environ.get("SKIN_XGBOOST_DIR", str(REPO_ROOT / "training" / "xgboost_fusion"))
).expanduser().resolve()

# Ensure the XGBoost artifacts directory is in sys.path for clinical_features import
if str(XGBOOST_DIR) not in sys.path:
    sys.path.insert(0, str(XGBOOST_DIR))


def model_paths() -> dict[str, Path]:
    return {
        "efficientnet": MODELS_DIR / "efficientnet_b0_best.keras",
        "clinical_xgboost": XGBOOST_DIR / "clinical_xgboost.json",
        "feature_schema": XGBOOST_DIR / "feature_schema.json",
        "training_config": XGBOOST_DIR / "training_config.json",
    }


class ModelRegistry:
    """Holds the active loaded models and clinical preprocessor.
    
    Active models:
      - EfficientNetB0 (Dermoscopic image model)
      - Clinical-Only XGBoost (53-feature clinical model)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Loaded model instances
        self.efficientnet: Any = None
        self.resnet: Any = None  # Explicitly None (disconnected)
        self.clinical_xgboost: Any = None
        self.clinical_preprocessor: Any = None

        # Artifacts and configuration
        self.feature_schema: Optional[Dict[str, Any]] = None
        self.training_config: Optional[Dict[str, Any]] = None

        # Pre-parsed feature lists
        self.feature_order: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.boolean_features: list[str] = []
        self.symptom_features: list[str] = []
        self.categorical_vocab: dict[str, list[str]] = {}

        # Dual-model status (no unvalidated ensemble)
        self.has_ensemble: bool = False
        self.weights: dict[str, float] = {}
        self.threshold: Optional[float] = None

        # Load state - tracks only actively loaded models/schemas
        self.loaded: dict[str, bool] = {
            "efficientnet_b0": False,
            "clinical_xgboost": False,
            "feature_schema": False,
        }
        self.errors: dict[str, str] = {}

    @property
    def ready(self) -> bool:
        return all(self.loaded.values())

    def inference_lock(self) -> threading.Lock:
        return self._lock

    def load_all(self) -> None:
        paths = model_paths()

        # 1. Feature Schema & Clinical Preprocessor
        try:
            schema_path = paths["feature_schema"]
            if not schema_path.is_file():
                raise FileNotFoundError(f"Missing schema: {schema_path}")
            self.feature_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self.feature_order = list(self.feature_schema.get("feature_order", []))
            self.numeric_features = list(self.feature_schema.get("numeric_features", {}).keys())
            self.boolean_features = list(self.feature_schema.get("boolean_features", {}).keys())
            self.symptom_features = list(self.feature_schema.get("symptom_features", {}).keys())

            cats = self.feature_schema.get("categorical_features", {})
            self.categorical_features = list(cats.keys())
            self.categorical_vocab = {
                k: list(v.get("categories_from_training", [])) for k, v in cats.items()
            }

            import clinical_features as cf
            self.clinical_preprocessor = cf.ClinicalPreprocessor.from_schema(self.feature_schema)
            self.loaded["feature_schema"] = True
        except Exception as exc:
            self.loaded["feature_schema"] = False
            self.errors["feature_schema"] = str(exc)

        # 2. Model 1: EfficientNetB0 (ISIC 2019)
        try:
            eff_path = paths["efficientnet"]
            if not eff_path.is_file():
                raise FileNotFoundError(f"Missing EfficientNetB0 model: {eff_path}")
            from tensorflow import keras
            self.efficientnet = keras.models.load_model(eff_path)
            self.loaded["efficientnet_b0"] = True
        except Exception as exc:
            self.loaded["efficientnet_b0"] = False
            self.errors["efficientnet_b0"] = str(exc)

        # 3. Model 2: Clinical-Only XGBoost (PAD-UFES-20)
        try:
            xgb_path = paths["clinical_xgboost"]
            if not xgb_path.is_file():
                raise FileNotFoundError(f"Missing Clinical XGBoost model: {xgb_path}")
            import xgboost as xgb
            self.clinical_xgboost = xgb.XGBClassifier()
            self.clinical_xgboost.load_model(str(xgb_path))
            self.loaded["clinical_xgboost"] = True
        except Exception as exc:
            self.loaded["clinical_xgboost"] = False
            self.errors["clinical_xgboost"] = str(exc)


registry = ModelRegistry()
