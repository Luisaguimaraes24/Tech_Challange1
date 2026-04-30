"""
src/data/preprocessing.py
Carregamento e limpeza do dataset Telco Customer Churn.

Responsabilidades:
    - Carregar o CSV bruto
    - Tratar TotalCharges (string com espaços em branco → float, imputar mediana)
    - Remover customerID (PII / sem poder preditivo)
    - Codificar target: Churn Yes=1 / No=0
    - Encoding binário: colunas Yes/No e Female/Male → 0/1
    - Salvar dataset limpo em data/processed/

Este módulo é importado tanto pelo script de EDA (notebooks/01_eda.py)
quanto pelo pipeline de treinamento (src/features/build_features.py).

Separação de responsabilidades:
    preprocessing.py  → limpeza + encoding binário simples (tipo de dado)
    build_features.py → One-Hot nas nominais + StandardScaler + split
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Colunas que o modelo nunca deve ver
COLUNAS_EXCLUIR = ["customerID"]

# Coluna alvo
TARGET = "Churn"

# Colunas numéricas contínuas (passam direto para o build_features)
COLUNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Colunas binárias simples Yes/No → 1/0
COLUNAS_BINARIAS_YES_NO = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Colunas com 3 categorias que incluem "No internet service" ou "No phone service"
# Mapeadas para ordinal: sem serviço=0, No=1, Yes=2
COLUNAS_SERVICOS = [
    "MultipleLines",      # No phone service / No / Yes
    "OnlineSecurity",     # No internet service / No / Yes
    "OnlineBackup",       # No internet service / No / Yes
    "DeviceProtection",   # No internet service / No / Yes
    "TechSupport",        # No internet service / No / Yes
    "StreamingTV",        # No internet service / No / Yes
    "StreamingMovies",    # No internet service / No / Yes
]

# Colunas nominais que ficam para One-Hot no build_features
COLUNAS_NOMINAIS = [
    "InternetService",   # DSL / Fiber optic / No
    "Contract",          # Month-to-month / One year / Two year
    "PaymentMethod",     # 4 categorias
]

# Mapeamento ordinal para colunas de serviço
_MAPA_SERVICO = {
    "No phone service":    0,
    "No internet service": 0,
    "No":                  1,
    "Yes":                 2,
}

# Mapeamento binário Yes/No
_MAPA_YES_NO = {"No": 0, "Yes": 1}

# Mapeamento gender
_MAPA_GENDER = {"Female": 0, "Male": 1}


def carregar_dados(caminho: str | Path) -> pd.DataFrame:
    """
    Carrega o CSV bruto e retorna o DataFrame sem nenhuma transformação.

    Args:
        caminho: caminho para o arquivo CSV.

    Returns:
        DataFrame com os dados brutos.
    """
    caminho = Path(caminho)
    logger.info("Carregando dataset: %s", caminho)
    df = pd.read_csv(caminho)
    logger.info("Shape bruto: %d linhas x %d colunas", *df.shape)
    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza e encoding básico ao dataset bruto:
      1. Remove customerID (PII)
      2. Converte TotalCharges para float e imputa mediana onde há espaços em branco
      3. Codifica target Churn: Yes=1 / No=0
      4. Encoding binário: gender (Female=0, Male=1)
      5. Encoding binário: colunas Yes/No simples → 0/1
      6. Encoding ordinal: colunas de serviço (sem serviço=0, No=1, Yes=2)

    As colunas nominais (InternetService, Contract, PaymentMethod) permanecem
    como texto — serão tratadas com One-Hot Encoding no build_features.py.

    Args:
        df: DataFrame bruto.

    Returns:
        DataFrame limpo e com encoding básico aplicado.
    """
    df_clean = df.copy()

    # 1. Remover identificador (sem poder preditivo e é PII)
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
        logger.info("customerID removido.")

    # 2. TotalCharges: espaços em branco → NaN → imputar com mediana
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    n_nulos = df_clean["TotalCharges"].isnull().sum()
    if n_nulos > 0:
        mediana = df_clean["TotalCharges"].median()
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(mediana)
        logger.info(
            "TotalCharges: %d registros com valor vazio imputados com mediana (%.2f).",
            n_nulos,
            mediana,
        )

    # 3. Codificar target
    df_clean[TARGET] = (df_clean[TARGET] == "Yes").astype(int)
    n_churn = df_clean[TARGET].sum()
    pct = n_churn / len(df_clean) * 100
    logger.info(
        "Target codificado — Churn=1: %d (%.1f%%) | Churn=0: %d (%.1f%%)",
        n_churn, pct,
        len(df_clean) - n_churn, 100 - pct,
    )

    # 4. Gender: Female=0 / Male=1
    if "gender" in df_clean.columns:
        df_clean["gender"] = df_clean["gender"].map(_MAPA_GENDER)
        logger.info("gender codificado: Female=0 / Male=1.")

    # 5. Colunas binárias Yes/No → 0/1
    for col in COLUNAS_BINARIAS_YES_NO:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(_MAPA_YES_NO)
    logger.info(
        "Encoding binário Yes/No aplicado em: %s.",
        ", ".join(COLUNAS_BINARIAS_YES_NO),
    )

    # 6. Colunas de serviço: sem serviço=0 / No=1 / Yes=2
    for col in COLUNAS_SERVICOS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(_MAPA_SERVICO)
    logger.info(
        "Encoding ordinal de serviços aplicado em: %s.",
        ", ".join(COLUNAS_SERVICOS),
    )

    # SeniorCitizen já é 0/1 — sem ação necessária
    logger.info("SeniorCitizen já é 0/1. Sem alteração.")

    logger.info("Shape limpo: %d linhas x %d colunas", *df_clean.shape)
    return df_clean


def salvar_dados_processados(df: pd.DataFrame, caminho: str | Path) -> None:
    """
    Salva o DataFrame limpo em CSV.

    Args:
        df: DataFrame limpo.
        caminho: caminho de destino.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    logger.info("Dataset processado salvo em: %s", caminho)


def carregar_dados_processados(caminho: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset já processado (data/processed/).

    Args:
        caminho: caminho para o CSV processado.

    Returns:
        DataFrame limpo.
    """
    caminho = Path(caminho)
    logger.info("Carregando dataset processado: %s", caminho)
    df = pd.read_csv(caminho)
    logger.info("Shape: %d linhas x %d colunas", *df.shape)
    return df