from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"

RESULTADOS_PATH = MODELS_DIR / "resultados_anomalias.csv"
MODELO_PATH = MODELS_DIR / "modelo_isolation_forest.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURES_PATH = MODELS_DIR / "features_modelo.pkl"

app = FastAPI(
    title="Metro Anomaly Dashboard API",
    description="API para dashboard de detecção de anomalias em tráfego veicular.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _safe_value(value: Any) -> Any:
    """Converte valores numpy/pandas para JSON seguro."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if pd.isna(value):
        return None

    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {col: _safe_value(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


def _parse_csv_list(value: str | None, cast_type=str) -> list[Any]:
    if not value:
        return []

    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(cast_type(item))
        except ValueError:
            continue

    return result


def gerar_resultados_se_necessario() -> None:
    """
    Gera resultados automaticamente se o CSV não existir.

    Para o trabalho, o recomendado é exportar o CSV pelo notebook e colocar em:
    models/resultados_anomalias.csv

    Esta função existe como plano B para quando o arquivo ainda não foi copiado.
    """
    if RESULTADOS_PATH.exists():
        return

    try:
        import joblib
        from ucimlrepo import fetch_ucirepo
        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "Arquivo models/resultados_anomalias.csv não encontrado e as dependências "
            "para gerar os resultados automaticamente não estão disponíveis."
        ) from exc

    metro_traffic = fetch_ucirepo(id=492)
    X_original = metro_traffic.data.features.copy()
    y_original = metro_traffic.data.targets.copy()

    if isinstance(y_original, pd.DataFrame):
        df = pd.concat([X_original, y_original], axis=1)
    else:
        df = X_original.copy()
        df["traffic_volume"] = y_original

    df_proc = df.copy().drop_duplicates().reset_index(drop=True)

    df_proc["date_time"] = pd.to_datetime(df_proc["date_time"], errors="coerce")
    df_proc["holiday"] = df_proc["holiday"].fillna("None")
    df_proc["weather_main"] = df_proc["weather_main"].fillna("Não informado")

    df_proc["is_holiday"] = np.where(
        df_proc["holiday"].astype(str).str.lower().eq("none"),
        0,
        1,
    )

    df_proc["year"] = df_proc["date_time"].dt.year
    df_proc["month"] = df_proc["date_time"].dt.month
    df_proc["day"] = df_proc["date_time"].dt.day
    df_proc["hour"] = df_proc["date_time"].dt.hour
    df_proc["day_of_week"] = df_proc["date_time"].dt.dayofweek
    df_proc["is_weekend"] = df_proc["day_of_week"].isin([5, 6]).astype(int)
    df_proc["temp_celsius"] = df_proc["temp"] - 273.15

    numeric_features = [
        "temp_celsius",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "traffic_volume",
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ]
    categorical_features = ["weather_main"]
    features_modelo = numeric_features + categorical_features

    X_modelo = df_proc[features_modelo].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_scaled = preprocessor.fit_transform(X_modelo)

    modelo_iforest = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )

    predicoes = modelo_iforest.fit_predict(X_scaled)

    df_resultado = df_proc.copy()
    df_resultado["predicao_iforest"] = predicoes
    df_resultado["anomalia"] = np.where(df_resultado["predicao_iforest"] == -1, 1, 0)
    df_resultado["classe"] = df_resultado["anomalia"].map({0: "Normal", 1: "Anomalia"})
    df_resultado["anomaly_score"] = -modelo_iforest.decision_function(X_scaled)

    matriz_pca = X_scaled.toarray() if hasattr(X_scaled, "toarray") else X_scaled
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(matriz_pca)
    df_resultado["pca_1"] = X_pca[:, 0]
    df_resultado["pca_2"] = X_pca[:, 1]

    referencia = (
        df_resultado.groupby(["day_of_week", "hour"])["traffic_volume"]
        .agg(["median", "mean"])
        .reset_index()
        .rename(
            columns={
                "median": "volume_mediano_contexto",
                "mean": "volume_medio_contexto",
            }
        )
    )

    df_resultado = df_resultado.merge(referencia, on=["day_of_week", "hour"], how="left")
    df_resultado["diferenca_para_mediana"] = (
        df_resultado["traffic_volume"] - df_resultado["volume_mediano_contexto"]
    )

    def interpretar_linha(row: pd.Series) -> str:
        motivos = []

        if row["traffic_volume"] < row["volume_mediano_contexto"] * 0.35:
            motivos.append("volume muito abaixo do esperado para esse dia/horário")
        elif row["traffic_volume"] > row["volume_mediano_contexto"] * 1.65:
            motivos.append("volume muito acima do esperado para esse dia/horário")

        if row["rain_1h"] > 0:
            motivos.append("presença de chuva no período")

        if row["snow_1h"] > 0:
            motivos.append("presença de neve no período")

        if row["is_holiday"] == 1:
            motivos.append("registro associado a feriado")

        if str(row["weather_main"]).lower() in ["thunderstorm", "squall", "fog", "mist", "snow"]:
            motivos.append(f"condição climática incomum: {row['weather_main']}")

        if not motivos:
            motivos.append("combinação incomum entre clima, horário e volume de tráfego")

        return "; ".join(motivos)

    df_resultado["interpretacao"] = np.where(
        df_resultado["anomalia"] == 1,
        df_resultado.apply(interpretar_linha, axis=1),
        "Registro considerado normal pelo modelo",
    )

    MODELS_DIR.mkdir(exist_ok=True)
    df_resultado.to_csv(RESULTADOS_PATH, index=False)
    joblib.dump(modelo_iforest, MODELO_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(features_modelo, FEATURES_PATH)


def carregar_dados() -> pd.DataFrame:
    gerar_resultados_se_necessario()

    if not RESULTADOS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Arquivo models/resultados_anomalias.csv não encontrado.",
        )

    df = pd.read_csv(RESULTADOS_PATH)
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    if "classe" not in df.columns and "anomalia" in df.columns:
        df["classe"] = df["anomalia"].map({0: "Normal", 1: "Anomalia"})

    if "temp_celsius" not in df.columns and "temp" in df.columns:
        df["temp_celsius"] = df["temp"] - 273.15

    if "is_holiday" not in df.columns and "holiday" in df.columns:
        df["is_holiday"] = np.where(
            df["holiday"].astype(str).str.lower().eq("none"),
            0,
            1,
        )

    if "volume_mediano_contexto" not in df.columns and {"day_of_week", "hour", "traffic_volume"}.issubset(df.columns):
        referencia = (
            df.groupby(["day_of_week", "hour"])["traffic_volume"]
            .median()
            .reset_index()
            .rename(columns={"traffic_volume": "volume_mediano_contexto"})
        )
        df = df.merge(referencia, on=["day_of_week", "hour"], how="left")

    if "diferenca_para_mediana" not in df.columns and {"traffic_volume", "volume_mediano_contexto"}.issubset(df.columns):
        df["diferenca_para_mediana"] = df["traffic_volume"] - df["volume_mediano_contexto"]

    return df


def aplicar_filtros(
    df: pd.DataFrame,
    months: str | None = None,
    weather: str | None = None,
    hour_min: int | None = None,
    hour_max: int | None = None,
    only_anomalies: bool = False,
    score_min: float | None = None,
) -> pd.DataFrame:
    filtrado = df.copy()

    meses = _parse_csv_list(months, int)
    if meses and "month" in filtrado.columns:
        filtrado = filtrado[filtrado["month"].isin(meses)]

    climas = _parse_csv_list(weather, str)
    if climas and "weather_main" in filtrado.columns:
        filtrado = filtrado[filtrado["weather_main"].isin(climas)]

    if hour_min is not None and "hour" in filtrado.columns:
        filtrado = filtrado[filtrado["hour"] >= hour_min]

    if hour_max is not None and "hour" in filtrado.columns:
        filtrado = filtrado[filtrado["hour"] <= hour_max]

    if only_anomalies and "anomalia" in filtrado.columns:
        filtrado = filtrado[filtrado["anomalia"] == 1]

    if score_min is not None and "anomaly_score" in filtrado.columns:
        filtrado = filtrado[filtrado["anomaly_score"] >= score_min]

    return filtrado


@app.get("/")
def pagina_inicial():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "dataset_disponivel": RESULTADOS_PATH.exists(),
        "modelo_disponivel": MODELO_PATH.exists(),
    }


@app.get("/api/dataset-info")
def dataset_info():
    df = carregar_dados()

    return {
        "dataset": "Metro Interstate Traffic Volume",
        "algoritmo": "Isolation Forest",
        "contamination": 0.03,
        "descricao": (
            "Dataset com registros horários de tráfego da rodovia I-94, contendo "
            "informações temporais, climáticas, feriados e volume de veículos."
        ),
        "total_registros": int(len(df)),
        "total_colunas": int(df.shape[1]),
        "colunas": list(df.columns),
        "periodo": {
            "inicio": _safe_value(df["date_time"].min()) if "date_time" in df.columns else None,
            "fim": _safe_value(df["date_time"].max()) if "date_time" in df.columns else None,
        },
    }


@app.get("/api/filters")
def filters():
    df = carregar_dados()

    return {
        "months": sorted([int(x) for x in df["month"].dropna().unique()]) if "month" in df.columns else [],
        "weather": sorted([str(x) for x in df["weather_main"].dropna().unique()]) if "weather_main" in df.columns else [],
        "hour_min": int(df["hour"].min()) if "hour" in df.columns else 0,
        "hour_max": int(df["hour"].max()) if "hour" in df.columns else 23,
        "score_min": float(df["anomaly_score"].min()) if "anomaly_score" in df.columns else 0.0,
        "score_max": float(df["anomaly_score"].max()) if "anomaly_score" in df.columns else 1.0,
    }


@app.get("/api/summary")
def summary(
    months: str | None = None,
    weather: str | None = None,
    hour_min: int | None = Query(default=None, ge=0, le=23),
    hour_max: int | None = Query(default=None, ge=0, le=23),
    only_anomalies: bool = False,
    score_min: float | None = None,
):
    df = carregar_dados()
    filtrado = aplicar_filtros(df, months, weather, hour_min, hour_max, only_anomalies, score_min)

    total = int(len(filtrado))
    total_anomalias = int(filtrado["anomalia"].sum()) if total and "anomalia" in filtrado.columns else 0
    percentual = round((total_anomalias / total * 100), 2) if total else 0.0

    return {
        "total_registros": total,
        "total_anomalias": total_anomalias,
        "percentual_anomalias": percentual,
        "media_volume": round(float(filtrado["traffic_volume"].mean()), 2) if total and "traffic_volume" in filtrado.columns else 0,
        "maior_score": round(float(filtrado["anomaly_score"].max()), 6) if total and "anomaly_score" in filtrado.columns else 0,
        "menor_score": round(float(filtrado["anomaly_score"].min()), 6) if total and "anomaly_score" in filtrado.columns else 0,
    }


@app.get("/api/charts")
def charts(
    months: str | None = None,
    weather: str | None = None,
    hour_min: int | None = Query(default=None, ge=0, le=23),
    hour_max: int | None = Query(default=None, ge=0, le=23),
    only_anomalies: bool = False,
    score_min: float | None = None,
):
    df = carregar_dados()
    filtrado = aplicar_filtros(df, months, weather, hour_min, hour_max, only_anomalies, score_min)

    if filtrado.empty:
        return {
            "daily_volume": [],
            "hourly_volume": [],
            "weather_counts": [],
            "score_histogram": [],
            "pca_points": [],
        }

    daily_volume = []
    if {"date_time", "traffic_volume", "anomalia"}.issubset(filtrado.columns):
        temp = filtrado.copy()
        temp["date"] = temp["date_time"].dt.date.astype(str)
        daily = (
            temp.groupby("date")
            .agg(volume_medio=("traffic_volume", "mean"), anomalias=("anomalia", "sum"))
            .reset_index()
            .sort_values("date")
        )
        daily_volume = _records(daily.tail(120))

    hourly_volume = []
    if {"hour", "traffic_volume", "anomalia"}.issubset(filtrado.columns):
        hourly = (
            filtrado.groupby("hour")
            .agg(volume_medio=("traffic_volume", "mean"), anomalias=("anomalia", "sum"))
            .reset_index()
            .sort_values("hour")
        )
        hourly_volume = _records(hourly)

    weather_counts = []
    if "weather_main" in filtrado.columns:
        weather_df = (
            filtrado["weather_main"]
            .value_counts()
            .reset_index()
        )
        weather_df.columns = ["weather_main", "quantidade"]
        weather_counts = _records(weather_df.head(12))

    score_histogram = []
    if "anomaly_score" in filtrado.columns:
        counts, edges = np.histogram(filtrado["anomaly_score"].dropna(), bins=20)
        score_histogram = [
            {
                "faixa": f"{edges[i]:.3f} a {edges[i + 1]:.3f}",
                "quantidade": int(counts[i]),
            }
            for i in range(len(counts))
        ]

    pca_points = []
    if {"pca_1", "pca_2", "classe"}.issubset(filtrado.columns):
        sample_size = min(1500, len(filtrado))
        sample = filtrado.sample(sample_size, random_state=42) if len(filtrado) > sample_size else filtrado
        cols = ["pca_1", "pca_2", "classe", "traffic_volume", "anomaly_score", "weather_main"]
        pca_points = _records(sample[cols])

    return {
        "daily_volume": daily_volume,
        "hourly_volume": hourly_volume,
        "weather_counts": weather_counts,
        "score_histogram": score_histogram,
        "pca_points": pca_points,
    }


@app.get("/api/anomalies")
def anomalies(
    months: str | None = None,
    weather: str | None = None,
    hour_min: int | None = Query(default=None, ge=0, le=23),
    hour_max: int | None = Query(default=None, ge=0, le=23),
    score_min: float | None = None,
    limit: int = Query(default=50, ge=1, le=300),
):
    df = carregar_dados()
    filtrado = aplicar_filtros(
        df,
        months=months,
        weather=weather,
        hour_min=hour_min,
        hour_max=hour_max,
        only_anomalies=True,
        score_min=score_min,
    )

    if filtrado.empty:
        return {"rows": []}

    colunas = [
        "date_time",
        "traffic_volume",
        "volume_mediano_contexto",
        "diferenca_para_mediana",
        "temp_celsius",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "weather_main",
        "is_holiday",
        "anomaly_score",
        "interpretacao",
    ]
    colunas = [col for col in colunas if col in filtrado.columns]

    ordenado = filtrado.sort_values("anomaly_score", ascending=False).head(limit)
    return {"rows": _records(ordenado[colunas])}


class PredictionInput(BaseModel):
    traffic_volume: float
    temp_celsius: float
    rain_1h: float = 0
    snow_1h: float = 0
    clouds_all: float = 0
    year: int
    month: int
    day: int
    hour: int
    day_of_week: int
    is_weekend: int
    is_holiday: int
    weather_main: str


@app.post("/api/predict")
def predict(payload: PredictionInput):
    """
    Endpoint opcional para classificar um novo registro.
    Funciona quando os arquivos .pkl gerados pelo notebook estiverem na pasta models/.
    Se os .pkl estiverem desatualizados (versão diferente do scikit-learn), regera automaticamente.
    """
    try:
        import joblib
    except Exception as exc:
        raise HTTPException(status_code=500, detail="joblib não está instalado.") from exc

    if not MODELO_PATH.exists() or not PREPROCESSOR_PATH.exists() or not FEATURES_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Arquivos do modelo não encontrados. Coloque em models/: "
                "modelo_isolation_forest.pkl, preprocessor.pkl e features_modelo.pkl."
            ),
        )

    try:
        modelo = joblib.load(MODELO_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        features = joblib.load(FEATURES_PATH)
    except (AttributeError, Exception) as exc:
        # Incompatibilidade de versão do scikit-learn: regera os .pkl a partir do CSV existente
        try:
            _regenerar_pkls(joblib)
            modelo = joblib.load(MODELO_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            features = joblib.load(FEATURES_PATH)
        except Exception as regen_exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Não foi possível carregar nem regenerar os arquivos do modelo. "
                    f"Erro original: {exc}. Erro ao regenerar: {regen_exc}"
                ),
            ) from regen_exc

    try:
        entrada = pd.DataFrame([payload.model_dump()])
        entrada = entrada[features]
        entrada_processada = preprocessor.transform(entrada)
        predicao = int(modelo.predict(entrada_processada)[0])
        score = float(-modelo.decision_function(entrada_processada)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao executar predição: {exc}") from exc

    return {
        "predicao": predicao,
        "classe": "Anomalia" if predicao == -1 else "Normal",
        "anomalia": 1 if predicao == -1 else 0,
        "anomaly_score": score,
    }


def _regenerar_pkls(joblib) -> None:
    """Regera os arquivos .pkl a partir do CSV já existente, usando a versão local do scikit-learn."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(RESULTADOS_PATH)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    if "temp_celsius" not in df.columns and "temp" in df.columns:
        df["temp_celsius"] = df["temp"] - 273.15
    if "is_holiday" not in df.columns and "holiday" in df.columns:
        df["is_holiday"] = np.where(df["holiday"].astype(str).str.lower().eq("none"), 0, 1)

    numeric_features = [
        "temp_celsius", "rain_1h", "snow_1h", "clouds_all", "traffic_volume",
        "year", "month", "day", "hour", "day_of_week", "is_weekend", "is_holiday",
    ]
    categorical_features = ["weather_main"]
    features_modelo = numeric_features + categorical_features

    cols_present = [c for c in features_modelo if c in df.columns]
    numeric_present = [c for c in numeric_features if c in df.columns]
    categorical_present = [c for c in categorical_features if c in df.columns]

    X = df[cols_present].copy()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_present),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_present),
        ]
    )
    X_scaled = preprocessor.fit_transform(X)
    modelo = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    modelo.fit(X_scaled)

    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(cols_present, FEATURES_PATH)
