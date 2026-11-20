from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES_A,
    FEATURES_B,
    METADATA_FILE,
    MODEL_A_FILE,
    MODEL_B_FILE,
    NUMERIC_FEATURES_A,
    NUMERIC_FEATURES_B,
    TARGET,
    TARGET_LABELS,
    TARGET_THRESHOLD,
)


def create_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_knn_pipeline(numeric_features: list[str], k: int, p: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", create_preprocessor(numeric_features)),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=k, p=p, weights="distance", n_jobs=-1),
            ),
        ]
    )


def chronological_split(data: pd.DataFrame):
    target_dates = np.sort(data["FECHA_OBJETIVO"].dropna().unique())
    train_position = max(1, int(len(target_dates) * 0.70))
    validation_position = max(train_position + 1, int(len(target_dates) * 0.85))
    train_end = pd.Timestamp(target_dates[train_position - 1])
    validation_end = pd.Timestamp(target_dates[validation_position - 1])

    train = data[data["FECHA_OBJETIVO"] <= train_end].copy()
    validation = data[
        (data["FECHA_OBJETIVO"] > train_end) & (data["FECHA_OBJETIVO"] <= validation_end)
    ].copy()
    test = data[data["FECHA_OBJETIVO"] > validation_end].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("No existen suficientes fechas para crear entrenamiento, validación y prueba.")
    return train, validation, test


def metric_row(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def select_hyperparameters(train, validation, features, numeric_features):
    preprocessor = create_preprocessor(numeric_features)
    train_matrix = preprocessor.fit_transform(train[features])
    validation_matrix = preprocessor.transform(validation[features])
    rows = []
    for k in [3, 5, 7, 9, 11, 15, 21]:
        for p, distance in [(1, "Manhattan"), (2, "Euclidiana")]:
            classifier = KNeighborsClassifier(n_neighbors=k, p=p, weights="distance", n_jobs=-1)
            classifier.fit(train_matrix, train[TARGET])
            prediction = classifier.predict(validation_matrix)
            metrics = metric_row(validation[TARGET], prediction)
            rows.append({"k": k, "p": p, "distancia": distance, **metrics})
    results = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1_macro"], ascending=False
    )
    return results.iloc[0].to_dict(), results


def evaluate_pipeline(pipeline, train_validation, test, features):
    pipeline.fit(train_validation[features], train_validation[TARGET])
    prediction = pipeline.predict(test[features])
    return metric_row(test[TARGET], prediction)


def train_and_save(data: pd.DataFrame, models_dir: str | Path | None = None) -> dict:
    models_dir = Path(models_dir) if models_dir else MODEL_A_FILE.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = chronological_split(data)

    best_a, search_a = select_hyperparameters(train, validation, FEATURES_A, NUMERIC_FEATURES_A)
    best_b, search_b = select_hyperparameters(train, validation, FEATURES_B, NUMERIC_FEATURES_B)
    train_validation = pd.concat([train, validation]).sort_values("FECHA_OBJETIVO")

    evaluation_a = create_knn_pipeline(NUMERIC_FEATURES_A, int(best_a["k"]), int(best_a["p"]))
    evaluation_b = create_knn_pipeline(NUMERIC_FEATURES_B, int(best_b["k"]), int(best_b["p"]))
    metrics_a = evaluate_pipeline(evaluation_a, train_validation, test, FEATURES_A)
    metrics_b = evaluate_pipeline(evaluation_b, train_validation, test, FEATURES_B)

    baseline = Pipeline(
        [
            ("preprocessing", create_preprocessor(NUMERIC_FEATURES_A)),
            ("classifier", DummyClassifier(strategy="most_frequent", random_state=42)),
        ]
    )
    metrics_baseline = evaluate_pipeline(baseline, train_validation, test, FEATURES_A)

    # Modelos de despliegue: se ajustan con todas las etiquetas disponibles después de evaluar.
    final_a = create_knn_pipeline(NUMERIC_FEATURES_A, int(best_a["k"]), int(best_a["p"]))
    final_b = create_knn_pipeline(NUMERIC_FEATURES_B, int(best_b["k"]), int(best_b["p"]))
    final_a.fit(data[FEATURES_A], data[TARGET])
    final_b.fit(data[FEATURES_B], data[TARGET])

    model_a_path = models_dir / MODEL_A_FILE.name
    model_b_path = models_dir / MODEL_B_FILE.name
    metadata_path = models_dir / METADATA_FILE.name
    joblib.dump(final_a, model_a_path, compress=3)
    joblib.dump(final_b, model_b_path, compress=3)

    metadata = {
        "proyecto": "Clasificación de variaciones de precios agrícolas en Piura",
        "modelo_despliegue": "KNN con variables climáticas",
        "umbral_porcentual": TARGET_THRESHOLD,
        "clases": TARGET_LABELS,
        "caracteristicas_modelo_a": FEATURES_A,
        "caracteristicas_modelo_b": FEATURES_B,
        "hiperparametros": {
            "modelo_a": {"k": int(best_a["k"]), "p": int(best_a["p"]), "distancia": best_a["distancia"]},
            "modelo_b": {"k": int(best_b["k"]), "p": int(best_b["p"]), "distancia": best_b["distancia"]},
        },
        "metricas_prueba": {
            "referencia": metrics_baseline,
            "modelo_a_sin_clima": metrics_a,
            "modelo_b_con_clima": metrics_b,
        },
        "particiones": {
            "entrenamiento": int(len(train)),
            "validacion": int(len(validation)),
            "prueba": int(len(test)),
            "fecha_objetivo_fin_entrenamiento": train["FECHA_OBJETIVO"].max().strftime("%Y-%m-%d"),
            "fecha_objetivo_fin_validacion": validation["FECHA_OBJETIVO"].max().strftime("%Y-%m-%d"),
            "fecha_objetivo_fin_prueba": test["FECHA_OBJETIVO"].max().strftime("%Y-%m-%d"),
        },
        "entrenamiento_final": {
            "filas": int(len(data)),
            "productos": sorted(data["PRODUCTO"].unique().tolist()),
            "fecha_maxima_con_etiqueta": data["FECHA_OBJETIVO"].max().strftime("%Y-%m-%d"),
        },
        "versiones": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "advertencia": (
            "Modelo académico con desempeño modesto. La salida es una clasificación experimental, "
            "no una recomendación de compra o venta ni evidencia causal."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    search_a.to_csv(models_dir / "busqueda_modelo_a.csv", index=False)
    search_b.to_csv(models_dir / "busqueda_modelo_b.csv", index=False)
    return metadata


def load_metadata(path: str | Path = METADATA_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

