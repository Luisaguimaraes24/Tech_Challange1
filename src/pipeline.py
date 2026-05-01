"""
src/pipeline.py
Pipeline sklearn completo: preprocessing + LogisticRegression Ridge L2.

Responsabilidades:
    - Construir o sklearn.Pipeline end-to-end (ColumnTransformer + LogReg)
    - Treinar e salvar o pipeline em models/churn_pipeline.joblib
    - Expor `carregar_pipeline()` para uso na API e nos testes

Uso (treino):
    python src/pipeline.py

Uso (inferência):
    from src.pipeline import carregar_pipeline
    pipeline = carregar_pipeline()
    proba = pipeline.predict_proba(df_input)[:, 1]
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import carregar_dados, limpar_dados
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SEED = 42

COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]
COLUNAS_NOMINAIS = ["InternetService", "Contract", "PaymentMethod"]
TARGET = "Churn"

MODEL_PATH = Path("models/churn_pipeline.joblib")
DATA_RAW = Path("data/raw/Telco-Customer-Churn.csv")

# Parâmetros do modelo escolhido na Etapa 2
# LogisticRegression + Ridge L2 (C=0.1) — melhor resultado líquido (R$ 230.340)
PARAMS_LOGREG = {
    "C": 0.1,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": SEED,
}


# ---------------------------------------------------------------------------
# Construção do pipeline
# ---------------------------------------------------------------------------

def construir_pipeline() -> Pipeline:
    """
    Constrói o sklearn.Pipeline completo:
      - ColumnTransformer: StandardScaler (numéricas) + OneHotEncoder (nominais)
      - LogisticRegression com Ridge L2 (melhor modelo da Etapa 2)

    Returns:
        Pipeline sklearn não treinado.
    """
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                COLUNAS_NUMERICAS,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                COLUNAS_NOMINAIS,
            ),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", LogisticRegression(**PARAMS_LOGREG)),
        ]
    )

    logger.info("Pipeline construído: ColumnTransformer + LogisticRegression Ridge L2 (C=0.1)")
    return pipeline


# ---------------------------------------------------------------------------
# Treino e persistência
# ---------------------------------------------------------------------------

def treinar_e_salvar(caminho_raw: Path = DATA_RAW, caminho_modelo: Path = MODEL_PATH) -> Pipeline:
    """
    Carrega o dataset bruto, limpa, treina o pipeline no conjunto completo
    e salva em disco.

    Args:
        caminho_raw:    caminho para o CSV bruto.
        caminho_modelo: caminho de destino do .joblib.

    Returns:
        Pipeline treinado.
    """
    logger.info("Iniciando treino do pipeline de produção...")

    # Dados
    df_raw = carregar_dados(caminho_raw)
    df_clean = limpar_dados(df_raw)

    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET].values

    logger.info("Treinando no dataset completo — %d amostras, %d features", *X.shape)

    # Treino
    pipeline = construir_pipeline()
    pipeline.fit(X, y)

    # Persistência
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho_modelo)
    logger.info("Pipeline salvo em: %s", caminho_modelo)

    return pipeline


# ---------------------------------------------------------------------------
# Carregamento (usado pela API e pelos testes)
# ---------------------------------------------------------------------------

def carregar_pipeline(caminho_modelo: Path = MODEL_PATH) -> Pipeline:
    """
    Carrega o pipeline serializado do disco.

    Args:
        caminho_modelo: caminho para o arquivo .joblib.

    Returns:
        Pipeline sklearn treinado.

    Raises:
        FileNotFoundError: se o arquivo não existir.
    """
    if not caminho_modelo.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{caminho_modelo}'. "
            "Execute 'python src/pipeline.py' para treinar e salvar o pipeline."
        )
    pipeline = joblib.load(caminho_modelo)
    logger.info("Pipeline carregado de: %s", caminho_modelo)
    return pipeline


# ---------------------------------------------------------------------------
# Inferência — função reutilizável
# ---------------------------------------------------------------------------

def prever(dados: dict | pd.DataFrame, pipeline: Pipeline | None = None) -> dict:
    """
    Realiza a inferência de churn para um ou mais clientes.

    Args:
        dados:    dict com os campos do cliente ou DataFrame já preparado.
        pipeline: pipeline carregado (se None, carrega do disco).

    Returns:
        dict com 'churn_prediction' (0 ou 1) e 'churn_probability' (float 0-1).
    """
    if pipeline is None:
        pipeline = carregar_pipeline()

    if isinstance(dados, dict):
        df = pd.DataFrame([dados])
    else:
        df = dados.copy()

    predicao = int(pipeline.predict(df)[0])
    probabilidade = float(pipeline.predict_proba(df)[0, 1])

    logger.info(
        "Predição — churn=%d | probabilidade=%.4f", predicao, probabilidade
    )

    return {
        "churn_prediction": predicao,
        "churn_probability": round(probabilidade, 4),
    }


# ---------------------------------------------------------------------------
# Entry point — treino
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    treinar_e_salvar()
    logger.info("Pipeline de produção pronto.")