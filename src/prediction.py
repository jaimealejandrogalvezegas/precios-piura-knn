from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES_B


def load_model(path: str | Path):
    return joblib.load(path)


def predict_row(model, row: pd.DataFrame) -> tuple[str, dict[str, float]]:
    missing = [column for column in FEATURES_B if column not in row.columns]
    if missing:
        raise ValueError(f"Faltan variables para predecir: {missing}")
    if row[FEATURES_B].isna().any().any():
        missing_values = row[FEATURES_B].columns[row[FEATURES_B].isna().any()].tolist()
        raise ValueError(f"La observación tiene valores faltantes en: {missing_values}")
    label = str(model.predict(row[FEATURES_B])[0])
    probabilities = model.predict_proba(row[FEATURES_B])[0]
    classes = model.named_steps["classifier"].classes_
    support = {str(name): float(value) for name, value in zip(classes, probabilities)}
    return label, support


def simple_explanation(row: pd.Series) -> list[str]:
    variation = float(row["VAR_PCT_ANT"])
    if variation > 1:
        price_text = f"El precio actual aumentó {variation:.1f} % respecto al registro anterior."
    elif variation < -1:
        price_text = f"El precio actual disminuyó {abs(variation):.1f} % respecto al registro anterior."
    else:
        price_text = f"El precio actual varió {variation:.1f} % respecto al registro anterior."
    return [
        price_text,
        f"El modelo utilizó {row['LLUVIA_ACUM_7D']:.1f} mm de lluvia acumulada en 7 días.",
        (
            f"También consideró temperaturas medias de {row['TMIN_MEDIA_7D']:.1f} °C "
            f"a {row['TMAX_MEDIA_7D']:.1f} °C durante los 7 días previos."
        ),
        f"El índice Niño 1+2 rezagado utilizado fue {row['NINO12']:.2f} °C.",
    ]

