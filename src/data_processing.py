from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    CLIMATE_FEATURES,
    CLIMATE_REQUIRED_COLUMNS,
    FEATURES_B,
    MARKET_REQUIRED_COLUMNS,
    SERIES_KEYS,
    TARGET,
    TARGET_THRESHOLD,
)


Source = str | Path | BinaryIO | TextIO


def _source_bytes(source: Source) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        value = source.getvalue()
        return value.encode("utf-8") if isinstance(value, str) else value
    position = source.tell() if hasattr(source, "tell") else None
    value = source.read()
    if position is not None and hasattr(source, "seek"):
        source.seek(position)
    return value.encode("utf-8") if isinstance(value, str) else value


def read_standard_csv(source: Source) -> pd.DataFrame:
    return pd.read_csv(BytesIO(_source_bytes(source)), skipinitialspace=True)


def read_nasa_power_csv(source: Source) -> pd.DataFrame:
    text = _source_bytes(source).decode("utf-8-sig")
    lines = text.splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.strip().startswith("YEAR,DOY"))
    except StopIteration as exc:
        raise ValueError("No se encontró la cabecera YEAR,DOY en el CSV de NASA POWER.") from exc
    return pd.read_csv(StringIO("\n".join(lines[header_index:])))


def validate_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name}: faltan las columnas obligatorias {missing}.")


def load_raw_sources(market_source: Source, climate_source: Source, nino_source: Source):
    market = read_standard_csv(market_source)
    climate = read_nasa_power_csv(climate_source)
    nino = read_standard_csv(nino_source)
    validate_columns(market, MARKET_REQUIRED_COLUMNS, "Mercado")
    validate_columns(climate, CLIMATE_REQUIRED_COLUMNS, "NASA POWER")
    if nino.shape[1] < 2:
        raise ValueError("NOAA Niño 1+2 debe contener una fecha y una columna numérica.")
    return market, climate, nino


def clean_market(market_raw: pd.DataFrame) -> pd.DataFrame:
    market = market_raw.copy()
    validate_columns(market, MARKET_REQUIRED_COLUMNS, "Mercado")
    market["FECHA"] = pd.to_datetime(
        market["FECHA_REGISTRO"].astype("string"), format="%Y%m%d", errors="coerce"
    )

    text_columns = [
        "PRODUCTO",
        "CATEGORIA",
        "UNIDAD_MEDIDA_MAY",
        "UNIDAD_MEDIDA_MIN",
    ]
    for column in text_columns:
        market[column] = (
            market[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    for column in ["PRECIO_MAYORISTA", "PRECIO_MINORISTA"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
        market.loc[market[column] <= 0, column] = np.nan

    wholesale = market[
        ["FECHA", "PRODUCTO", "CATEGORIA", "PRECIO_MAYORISTA", "UNIDAD_MEDIDA_MAY"]
    ].rename(
        columns={"PRECIO_MAYORISTA": "PRECIO", "UNIDAD_MEDIDA_MAY": "UNIDAD_MEDIDA"}
    )
    wholesale["TIPO_PRECIO"] = "MAYORISTA"

    retail = market[
        ["FECHA", "PRODUCTO", "CATEGORIA", "PRECIO_MINORISTA", "UNIDAD_MEDIDA_MIN"]
    ].rename(
        columns={"PRECIO_MINORISTA": "PRECIO", "UNIDAD_MEDIDA_MIN": "UNIDAD_MEDIDA"}
    )
    retail["TIPO_PRECIO"] = "MINORISTA"

    prices = pd.concat([wholesale, retail], ignore_index=True)
    prices = prices.dropna(subset=["FECHA", "PRODUCTO", "CATEGORIA", "PRECIO", "UNIDAD_MEDIDA"])
    prices = prices.drop_duplicates(SERIES_KEYS + ["FECHA"], keep="last")
    return prices.sort_values(SERIES_KEYS + ["FECHA"]).reset_index(drop=True)


def add_price_features(prices: pd.DataFrame, threshold: float = TARGET_THRESHOLD) -> pd.DataFrame:
    data = prices.sort_values(SERIES_KEYS + ["FECHA"]).copy()
    grouped = data.groupby(SERIES_KEYS, sort=False, group_keys=False)

    data["PRECIO_ACTUAL"] = data["PRECIO"]
    data["PRECIO_LAG1"] = grouped["PRECIO"].shift(1)
    data["PRECIO_LAG2"] = grouped["PRECIO"].shift(2)
    data["VAR_PCT_ANT"] = (data["PRECIO_ACTUAL"] / data["PRECIO_LAG1"] - 1) * 100
    data["MEDIA_MOVIL_3"] = grouped["PRECIO"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["STD_MOVIL_3"] = grouped["PRECIO"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).std()
    )
    data["DIAS_DESDE_ANT"] = grouped["FECHA"].diff().dt.days
    data["MES"] = data["FECHA"].dt.month
    data["DIA_SEMANA"] = data["FECHA"].dt.dayofweek

    data["FECHA_OBJETIVO"] = grouped["FECHA"].shift(-1)
    data["PRECIO_SIGUIENTE"] = grouped["PRECIO"].shift(-1)
    data["DIAS_HASTA_OBJETIVO"] = (data["FECHA_OBJETIVO"] - data["FECHA"]).dt.days
    data["VAR_SIG_PCT"] = (data["PRECIO_SIGUIENTE"] / data["PRECIO_ACTUAL"] - 1) * 100

    data[TARGET] = pd.Series(pd.NA, index=data.index, dtype="string")
    has_target = data["VAR_SIG_PCT"].notna()
    data.loc[has_target & (data["VAR_SIG_PCT"] > threshold), TARGET] = "SUBE"
    data.loc[has_target & (data["VAR_SIG_PCT"] < -threshold), TARGET] = "BAJA"
    data.loc[
        has_target & data["VAR_SIG_PCT"].between(-threshold, threshold, inclusive="both"),
        TARGET,
    ] = "ESTABLE"
    return data


def prepare_climate(climate_raw: pd.DataFrame) -> pd.DataFrame:
    climate = climate_raw.copy()
    validate_columns(climate, CLIMATE_REQUIRED_COLUMNS, "NASA POWER")
    climate["FECHA_CLIMA"] = pd.to_datetime(
        climate["YEAR"].astype("string") + "-" + climate["DOY"].astype("string"),
        format="%Y-%j",
        errors="coerce",
    )
    for column in ["PRECTOTCORR", "T2M_MAX", "T2M_MIN"]:
        climate[column] = pd.to_numeric(climate[column], errors="coerce").replace(-999, np.nan)

    climate = climate.dropna(subset=["FECHA_CLIMA"]).sort_values("FECHA_CLIMA")
    climate = climate.drop_duplicates("FECHA_CLIMA", keep="last")
    climate["LLUVIA_DIA_ANT"] = climate["PRECTOTCORR"]
    climate["TMAX_DIA_ANT"] = climate["T2M_MAX"]
    climate["TMIN_DIA_ANT"] = climate["T2M_MIN"]
    climate["LLUVIA_ACUM_7D"] = climate["PRECTOTCORR"].rolling(7, min_periods=1).sum()
    climate["LLUVIA_ACUM_30D"] = climate["PRECTOTCORR"].rolling(30, min_periods=1).sum()
    climate["TMAX_MEDIA_7D"] = climate["T2M_MAX"].rolling(7, min_periods=1).mean()
    climate["TMIN_MEDIA_7D"] = climate["T2M_MIN"].rolling(7, min_periods=1).mean()
    climate["FECHA_DISP_CLIMA"] = climate["FECHA_CLIMA"] + np.timedelta64(1, "D")
    return climate


def prepare_nino(nino_raw: pd.DataFrame) -> pd.DataFrame:
    nino = nino_raw.iloc[:, :2].copy()
    nino.columns = ["FECHA_NINO", "NINO12"]
    nino["FECHA_NINO"] = pd.to_datetime(nino["FECHA_NINO"], errors="coerce")
    nino["NINO12"] = pd.to_numeric(nino["NINO12"], errors="coerce").replace(-99.99, np.nan)
    nino = nino.dropna(subset=["FECHA_NINO", "NINO12"]).sort_values("FECHA_NINO")
    nino = nino.drop_duplicates("FECHA_NINO", keep="last")
    nino["FECHA_DISP_NINO"] = nino["FECHA_NINO"] + pd.offsets.MonthBegin(1)
    return nino


def consolidate_sources(
    market_raw: pd.DataFrame,
    climate_raw: pd.DataFrame,
    nino_raw: pd.DataFrame,
    threshold: float = TARGET_THRESHOLD,
) -> pd.DataFrame:
    prices = add_price_features(clean_market(market_raw), threshold=threshold)
    climate = prepare_climate(climate_raw)
    nino = prepare_nino(nino_raw)

    climate_columns = ["FECHA_CLIMA", "FECHA_DISP_CLIMA"] + CLIMATE_FEATURES
    consolidated = pd.merge_asof(
        prices.sort_values("FECHA"),
        climate[climate_columns].sort_values("FECHA_DISP_CLIMA"),
        left_on="FECHA",
        right_on="FECHA_DISP_CLIMA",
        direction="backward",
    )
    consolidated = pd.merge_asof(
        consolidated.sort_values("FECHA"),
        nino[["FECHA_NINO", "FECHA_DISP_NINO", "NINO12"]].sort_values("FECHA_DISP_NINO"),
        left_on="FECHA",
        right_on="FECHA_DISP_NINO",
        direction="backward",
    )
    return consolidated.sort_values(SERIES_KEYS + ["FECHA"]).reset_index(drop=True)


def model_ready_rows(consolidated: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES_B + [TARGET, "FECHA", "FECHA_OBJETIVO"]
    result = consolidated.dropna(subset=required).copy()
    result[TARGET] = result[TARGET].astype(str)
    return result.sort_values(["FECHA_OBJETIVO", "FECHA"]).reset_index(drop=True)


def prediction_ready_rows(consolidated: pd.DataFrame) -> pd.DataFrame:
    return consolidated.dropna(subset=FEATURES_B + ["FECHA"]).copy()


def quality_report(consolidated: pd.DataFrame) -> dict:
    ready = model_ready_rows(consolidated)
    climate_violations = int(
        ((consolidated["FECHA_CLIMA"].notna()) & (consolidated["FECHA_CLIMA"] >= consolidated["FECHA"])).sum()
    )
    month_start = consolidated["FECHA"].dt.to_period("M").dt.start_time
    nino_violations = int(
        ((consolidated["FECHA_NINO"].notna()) & (consolidated["FECHA_NINO"] >= month_start)).sum()
    )
    return {
        "filas_consolidadas": int(len(consolidated)),
        "filas_modelables": int(len(ready)),
        "productos": int(consolidated["PRODUCTO"].nunique()),
        "series": int(consolidated[SERIES_KEYS].drop_duplicates().shape[0]),
        "fecha_minima": consolidated["FECHA"].min(),
        "fecha_maxima": consolidated["FECHA"].max(),
        "duplicados_serie_fecha": int(consolidated.duplicated(SERIES_KEYS + ["FECHA"]).sum()),
        "violaciones_clima_futuro": climate_violations,
        "violaciones_nino_desfase": nino_violations,
    }


def load_processed(source: Source) -> pd.DataFrame:
    data = read_standard_csv(source)
    for column in ["FECHA", "FECHA_OBJETIVO", "FECHA_CLIMA", "FECHA_NINO"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def save_processed(data: pd.DataFrame, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False, date_format="%Y-%m-%d")

