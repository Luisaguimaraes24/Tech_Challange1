"""
tests/test_schema.py
Testes de schema com Pandera — Telco Customer Churn

Valida que o dataset processado (telco_clean.csv) obedece ao contrato
de dados esperado pelo pipeline de ML.

Referência: Eng. Software, Aula 03 — Testes Automatizados
"""

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest
from pandera import Column, DataFrameSchema

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Schema esperado após preprocessing.limpar_dados()
# ---------------------------------------------------------------------------

SCHEMA_TELCO_CLEAN = DataFrameSchema(
    columns={
        # Demográficos
        "gender": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "SeniorCitizen": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Partner": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Dependents": Column(int, pa.Check.isin([0, 1]), nullable=False),

        # Conta
        "tenure": Column(int, pa.Check.greater_than_or_equal_to(0), nullable=False),
        "PhoneService": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "MultipleLines": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),

        # Internet
        "InternetService": Column(str, nullable=False),
        "OnlineSecurity": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),
        "OnlineBackup": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),
        "DeviceProtection": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),
        "TechSupport": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),
        "StreamingTV": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),
        "StreamingMovies": Column(int, pa.Check.isin([0, 1, 2]), nullable=False),

        # Contrato e pagamento
        "Contract": Column(str, nullable=False),
        "PaperlessBilling": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "PaymentMethod": Column(str, nullable=False),

        # Financeiro
        "MonthlyCharges": Column(float, pa.Check.greater_than(0), nullable=False),
        "TotalCharges": Column(float, pa.Check.greater_than_or_equal_to(0), nullable=False),

        # Target
        "Churn": Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    checks=[
        # Sem valores nulos em nenhuma coluna
        pa.Check(lambda df: df.isnull().sum().sum() == 0, error="Dataset contém valores nulos"),
    ],
    strict=False,  # Permite colunas extras sem falhar
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_processado():
    """Carrega o dataset limpo para os testes de schema."""
    caminho = Path("data/processed/telco_clean.csv")
    if not caminho.exists():
        pytest.skip(
            "telco_clean.csv não encontrado. Execute 'python notebooks/01_eda.py' primeiro."
        )
    return pd.read_csv(caminho)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_schema_dataset_processado(df_processado):
    """O dataset processado deve obedecer ao schema definido."""
    SCHEMA_TELCO_CLEAN.validate(df_processado)


def test_sem_valores_nulos(df_processado):
    """Não deve haver valores nulos após o preprocessing."""
    nulos = df_processado.isnull().sum()
    colunas_com_nulos = nulos[nulos > 0]
    assert colunas_com_nulos.empty, (
        f"Colunas com valores nulos encontradas:\n{colunas_com_nulos}"
    )


def test_target_binario(df_processado):
    """A coluna Churn deve conter apenas 0 e 1."""
    valores_unicos = set(df_processado["Churn"].unique())
    assert valores_unicos == {0, 1}, (
        f"Churn deve conter apenas {{0, 1}}, mas contém: {valores_unicos}"
    )


def test_volume_minimo(df_processado):
    """O dataset deve ter pelo menos 7000 registros."""
    assert len(df_processado) >= 7000, (
        f"Dataset com apenas {len(df_processado)} registros — esperado >= 7000"
    )


def test_customer_id_removido(df_processado):
    """customerID deve ter sido removido (é PII)."""
    assert "customerID" not in df_processado.columns, (
        "customerID ainda presente no dataset processado — é PII e deve ser removido"
    )


def test_total_charges_float(df_processado):
    """TotalCharges deve ser float (não string)."""
    assert pd.api.types.is_float_dtype(df_processado["TotalCharges"]), (
        "TotalCharges deve ser float após conversão"
    )


def test_churn_desbalanceado_esperado(df_processado):
    """Taxa de churn deve estar entre 20% e 35% (conforme EDA)."""
    taxa_churn = df_processado["Churn"].mean()
    assert 0.20 <= taxa_churn <= 0.35, (
        f"Taxa de churn inesperada: {taxa_churn:.2%} (esperado 20%–35%)"
    )