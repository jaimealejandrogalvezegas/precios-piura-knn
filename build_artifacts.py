"""Genera la base consolidada y entrena los modelos reproducibles del proyecto."""

from src.config import CLIMATE_FILE, MARKET_FILE, NINO_FILE, PROCESSED_FILE
from src.data_processing import (
    consolidate_sources,
    load_raw_sources,
    model_ready_rows,
    quality_report,
    save_processed,
)
from src.modeling import train_and_save


def main():
    market, climate, nino = load_raw_sources(MARKET_FILE, CLIMATE_FILE, NINO_FILE)
    consolidated = consolidate_sources(market, climate, nino)
    report = quality_report(consolidated)
    save_processed(consolidated, PROCESSED_FILE)
    metadata = train_and_save(model_ready_rows(consolidated))

    print("Artefactos generados correctamente")
    for key, value in report.items():
        print(f"- {key}: {value}")
    metrics = metadata["metricas_prueba"]["modelo_b_con_clima"]
    print(f"- balanced_accuracy prueba: {metrics['balanced_accuracy']:.3f}")
    print(f"- f1_macro prueba: {metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
