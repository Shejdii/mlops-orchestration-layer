import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "regime_classifier.pkl"
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "test.csv"


def fail(message: str) -> None:
    print(json.dumps({"success": False, "message": message}, indent=2))
    sys.exit(1)


def main() -> None:
    if not MODEL_PATH.exists():
        fail(f"Model artifact not found: {MODEL_PATH}")

    if not INPUT_PATH.exists():
        fail(f"Smoke input CSV not found: {INPUT_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        fail(f"Failed to load model artifact: {exc}")

    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as exc:
        fail(f"Failed to load smoke input CSV: {exc}")

    if df.empty:
        fail("Smoke input CSV is empty.")

    sample = df.head(1)

    try:
        prediction = model.predict(sample)
    except Exception as exc:
        fail(f"Prediction failed: {exc}")

    print(
        json.dumps(
            {
                "success": True,
                "message": "Smoke prediction passed.",
                "rows_used": len(sample),
                "prediction_preview": (
                    prediction.tolist() if hasattr(prediction, "tolist") else str(prediction)
                ),
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
