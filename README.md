# Clasificación de variaciones de precios agrícolas en Piura

Proyecto académico reproducible que utiliza **K-Nearest Neighbors (KNN)** para clasificar si el precio mayorista o minorista de un producto del Mercado Modelo de Piura podría **SUBIR**, **BAJAR** o permanecer **ESTABLE** en su siguiente registro disponible.

La aplicación combina historial de precios, precipitación, temperatura e índice Niño 1+2. No calcula el precio exacto, la demanda, la rentabilidad ni qué producto conviene vender.

## Aplicación Streamlit

El prototipo permite:

- seleccionar un producto agrícola;
- comparar precios mayoristas y minoristas;
- visualizar clima histórico;
- obtener una clasificación experimental `SUBE`, `BAJA` o `ESTABLE`;
- consultar una explicación sencilla de las variables utilizadas;
- cargar CSV compatibles de forma opcional;
- descargar una base cargada después de validarla y consolidarla.

La aplicación no genera clasificaciones cuando faltan columnas, historial, clima o cuando el producto no estuvo presente durante el entrenamiento.

## Resultados del experimento

Los resultados corresponden al bloque cronológico final de prueba:

| Modelo | Accuracy | Balanced accuracy | F1 macro |
|---|---:|---:|---:|
| Referencia: clase mayoritaria | 0.675 | 0.333 | 0.269 |
| KNN sin clima | 0.632 | 0.355 | 0.338 |
| KNN con clima | 0.585 | 0.358 | 0.353 |

El modelo con clima presenta una mejora numérica pequeña en balanced accuracy y F1 macro, pero no constituye evidencia de una mejora robusta ni de causalidad.

## Estructura del repositorio

```text
precios-piura-knn/
├── app.py                       # Aplicación Streamlit
├── build_artifacts.py           # Consolida datos y vuelve a entrenar
├── requirements.txt             # Dependencias de ejecución
├── requirements-dev.txt         # Dependencias para pruebas
├── data/
│   ├── raw/                     # Tres CSV originales
│   └── processed/               # Base consolidada
├── models/                      # Pipelines KNN y metadatos
├── notebook/                    # Notebook académico ejecutado
├── src/                         # Limpieza, integración y modelado
├── tests/                       # Pruebas de datos y artefactos
└── .streamlit/config.toml       # Tema institucional #E26400
```

## Instalación local

Se recomienda Python 3.11, 3.12 o 3.13.

```bash
git clone URL_DEL_REPOSITORIO
cd precios-piura-knn
python -m venv .venv
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En macOS o Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ejecute la aplicación:

```bash
streamlit run app.py
```

## Reproducir la consolidación y el entrenamiento

Los modelos ya están incluidos. Para regenerar la base procesada y volver a entrenar desde los CSV originales:

```bash
python build_artifacts.py
```

Este comando:

1. valida las tres fuentes;
2. limpia los precios no positivos;
3. conserva por separado mayorista, minorista y unidad;
4. crea variables históricas por serie;
5. integra clima hasta el día anterior;
6. integra Niño 1+2 con un mes de desfase;
7. divide cronológicamente entrenamiento, validación y prueba;
8. selecciona K y distancia mediante validación;
9. guarda los pipelines completos en `models/`.

## Ejecutar las pruebas

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Las pruebas comprueban dimensiones, duplicados, variables completas, desfases temporales y que el modelo pueda producir una clasificación válida.

## Archivos compatibles para carga opcional

### Mercado

Debe contener como mínimo:

```text
FECHA_REGISTRO
PRODUCTO
PRECIO_MAYORISTA
PRECIO_MINORISTA
CATEGORIA
UNIDAD_MEDIDA_MAY
UNIDAD_MEDIDA_MIN
```

`FECHA_REGISTRO` debe utilizar el formato `AAAAMMDD`. Se requieren al menos tres precios válidos por producto, tipo y unidad para construir los rezagos.

### NASA POWER

Debe contener:

```text
YEAR, DOY, PRECTOTCORR, T2M_MAX, T2M_MIN
```

### Niño 1+2

Debe contener una columna de fecha y una columna numérica con la anomalía mensual.

Si el usuario carga solamente el mercado, la aplicación utiliza los archivos NASA y NOAA incluidos, siempre que las nuevas fechas estén cubiertas.

## Despliegue en Streamlit Community Cloud

1. Publique esta carpeta como repositorio en GitHub.
2. Ingrese a Streamlit Community Cloud.
3. Seleccione **Create app** y conecte el repositorio.
4. Seleccione la rama `main`.
5. Indique `app.py` como archivo principal.
6. Inicie el despliegue.

No se necesitan claves ni secretos para esta versión.

## Fuentes de datos

- [Gobierno Regional Piura: precios mayoristas y minoristas](https://www.datosabiertos.gob.pe/dataset/precios-mayorista-y-minorista-del-mercado-modelo-de-piuragobierno-regional-piura-grp-0)
- [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- [NOAA PSL: índices mensuales](https://psl.noaa.gov/data/timeseries/month/)

## Limitaciones y uso responsable

- La clase corresponde al siguiente registro disponible, no a una cantidad fija de días.
- La clase `ESTABLE` es mayoritaria.
- NASA POWER representa una celda geográfica y no una estación dentro del mercado.
- No se incluyen ventas, demanda, oferta, márgenes, transporte ni volumen comercializado.
- Las proporciones de vecinos KNN no son probabilidades calibradas.
- Una asociación entre clima y precios no demuestra causalidad.
- El prototipo no debe utilizarse como recomendación financiera o comercial.

## Licencia

El código se distribuye bajo la licencia MIT. Los datos conservan las condiciones de sus fuentes originales.
