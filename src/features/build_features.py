"""
src/features/build_features.py
Feature engineering e preparação final para modelagem.

Responsabilidades:
    - One-Hot Encoding nas colunas nominais (InternetService, Contract, PaymentMethod)
    - StandardScaler nas colunas numéricas contínuas
    - Train/test split estratificado (preserva proporção de churn)
    - Retorna X_train, X_test, y_train, y_test prontos para o modelo

Separação de responsabilidades:
    preprocessing.py  → limpeza + encoding binário/ordinal
    build_features.py → One-Hot + StandardScaler + split  ← este módulo
    02_baselines.py   → importa daqui e treina os modelos
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Seed global
SEED = 42

# Colunas numéricas que precisam de escala
COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Colunas nominais que recebem One-Hot Encoding
COLUNAS_NOMINAIS = ["InternetService", "Contract", "PaymentMethod"]

# Coluna alvo
TARGET = "Churn"


def construir_pipeline_preprocessamento() -> ColumnTransformer:
    """
    Constrói o ColumnTransformer com:
      - StandardScaler nas numéricas
      - OneHotEncoder nas nominais (drop='first' para evitar multicolinearidade)

    Returns:
        ColumnTransformer configurado (não fitado).
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
        remainder="passthrough",  # colunas já numéricas (binárias/ordinais) passam direto
    )
    return preprocessador


def preparar_dados(
    caminho_csv: str | Path,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Carrega o dataset processado, aplica One-Hot + StandardScaler
    e retorna os conjuntos de treino e teste estratificados.

    Args:
        caminho_csv: caminho para o telco_clean.csv (data/processed/).
        test_size:   proporção do conjunto de teste (padrão: 20%).

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    caminho_csv = Path(caminho_csv)
    logger.info("Carregando dataset processado: %s", caminho_csv)
    df = pd.read_csv(caminho_csv)
    logger.info("Shape: %d linhas x %d colunas", *df.shape)

    # Separar features e target
    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    logger.info("Features: %d | Target: Churn (0/1)", X.shape[1])
    logger.info("Distribuição do target — 0: %d | 1: %d", (y == 0).sum(), (y == 1).sum())

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )
    logger.info(
        "Split estratificado — treino: %d | teste: %d (test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )
    logger.info(
        "Churn no treino: %.2f%% | no teste: %.2f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    # Preprocessamento: fit no treino, transform em ambos
    preprocessador = construir_pipeline_preprocessamento()
    X_train_proc = preprocessador.fit_transform(X_train)
    X_test_proc = preprocessador.transform(X_test)

    # Recuperar nomes das features após transformação
    nomes_num = COLUNAS_NUMERICAS
    nomes_ohe = (
        preprocessador
        .named_transformers_["cat"]
        .get_feature_names_out(COLUNAS_NOMINAIS)
        .tolist()
    )
    colunas_passthrough = [
        c for c in X.columns
        if c not in COLUNAS_NUMERICAS + COLUNAS_NOMINAIS
    ]
    feature_names = nomes_num + nomes_ohe + colunas_passthrough

    logger.info(
        "Shape final — X_train: %s | X_test: %s | features: %d",
        X_train_proc.shape, X_test_proc.shape, len(feature_names),
    )

    return X_train_proc, X_test_proc, y_train, y_test, feature_names