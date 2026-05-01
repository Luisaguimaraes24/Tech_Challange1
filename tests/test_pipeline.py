"""
tests/test_pipeline.py
Testes unitários — preprocessing e pipeline de ML

Cobre:
    - limpar_dados(): conversão de tipos, encoding, remoção de PII
    - construir_pipeline(): estrutura do sklearn.Pipeline
    - prever(): formato da saída de inferência

Referência: Eng. Software, Aula 03 — Testes Automatizados
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import limpar_dados
from src.pipeline import construir_pipeline, prever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_raw_mock():
    """
    DataFrame mínimo simulando o CSV bruto do Telco.
    Contém 5 clientes com valores representativos do dataset real.
    """
    return pd.DataFrame(
        {
            "customerID":        ["001-AAA", "002-BBB", "003-CCC", "004-DDD", "005-EEE"],
            "gender":            ["Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen":     [0, 1, 0, 0, 1],
            "Partner":           ["Yes", "No", "Yes", "No", "Yes"],
            "Dependents":        ["No", "No", "Yes", "No", "No"],
            "tenure":            [12, 1, 48, 5, 72],
            "PhoneService":      ["Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines":     ["No", "No", "No phone service", "Yes", "No"],
            "InternetService":   ["Fiber optic", "DSL", "DSL", "Fiber optic", "No"],
            "OnlineSecurity":    ["No", "Yes", "No", "No", "No internet service"],
            "OnlineBackup":      ["Yes", "No", "Yes", "No", "No internet service"],
            "DeviceProtection":  ["No", "Yes", "No", "Yes", "No internet service"],
            "TechSupport":       ["No", "No", "No", "No", "No internet service"],
            "StreamingTV":       ["No", "No", "No", "Yes", "No internet service"],
            "StreamingMovies":   ["No", "No", "No", "No", "No internet service"],
            "Contract":          ["Month-to-month", "Month-to-month", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling":  ["Yes", "Yes", "No", "Yes", "No"],
            "PaymentMethod":     [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges":    [85.7, 53.85, 42.30, 70.70, 20.00],
            "TotalCharges":      ["1028.40", "53.85", "2030.40", "353.50", " "],  # espaço = NaN
            "Churn":             ["Yes", "No", "No", "Yes", "No"],
        }
    )


@pytest.fixture
def df_limpo(df_raw_mock):
    """DataFrame já passado por limpar_dados()."""
    return limpar_dados(df_raw_mock)


@pytest.fixture
def pipeline_nao_treinado():
    """Pipeline sklearn sem fit."""
    return construir_pipeline()


# ---------------------------------------------------------------------------
# Testes — limpar_dados()
# ---------------------------------------------------------------------------

class TestLimparDados:
    def test_remove_customer_id(self, df_limpo):
        """customerID deve ser removido (PII)."""
        assert "customerID" not in df_limpo.columns

    def test_total_charges_eh_float(self, df_limpo):
        """TotalCharges deve ser float após conversão."""
        assert pd.api.types.is_float_dtype(df_limpo["TotalCharges"])

    def test_total_charges_sem_nulos(self, df_limpo):
        """Espaços em branco em TotalCharges devem ser imputados (sem NaN)."""
        assert df_limpo["TotalCharges"].isnull().sum() == 0

    def test_churn_binario(self, df_limpo):
        """Churn deve ser 0 ou 1 (não 'Yes'/'No')."""
        valores = set(df_limpo["Churn"].unique())
        assert valores.issubset({0, 1})

    def test_churn_yes_vira_1(self, df_raw_mock, df_limpo):
        """Clientes com Churn='Yes' no raw devem ter Churn=1 no limpo."""
        idx_churn_yes = df_raw_mock[df_raw_mock["Churn"] == "Yes"].index
        assert all(df_limpo.loc[idx_churn_yes, "Churn"] == 1)

    def test_gender_binario(self, df_limpo):
        """gender deve ser 0 (Female) ou 1 (Male)."""
        assert set(df_limpo["gender"].unique()).issubset({0, 1})

    def test_partner_binario(self, df_limpo):
        """Partner deve ser 0 ou 1."""
        assert set(df_limpo["Partner"].unique()).issubset({0, 1})

    def test_multiple_lines_ordinal(self, df_limpo):
        """MultipleLines deve ser 0, 1 ou 2."""
        assert set(df_limpo["MultipleLines"].unique()).issubset({0, 1, 2})

    def test_sem_nulos_apos_limpeza(self, df_limpo):
        """Nenhuma coluna deve ter valores nulos após limpar_dados()."""
        nulos = df_limpo.isnull().sum().sum()
        assert nulos == 0, f"{nulos} valores nulos encontrados após limpeza"

    def test_shape_preservado(self, df_raw_mock, df_limpo):
        """Número de linhas deve ser o mesmo; colunas reduzem em 1 (customerID)."""
        assert len(df_limpo) == len(df_raw_mock)
        assert df_limpo.shape[1] == df_raw_mock.shape[1] - 1  # sem customerID


# ---------------------------------------------------------------------------
# Testes — construir_pipeline()
# ---------------------------------------------------------------------------

class TestConstruirPipeline:
    def test_retorna_sklearn_pipeline(self, pipeline_nao_treinado):
        """construir_pipeline() deve retornar um sklearn.Pipeline."""
        assert isinstance(pipeline_nao_treinado, Pipeline)

    def test_tem_dois_steps(self, pipeline_nao_treinado):
        """Pipeline deve ter exatamente 2 steps: preprocessamento + modelo."""
        assert len(pipeline_nao_treinado.steps) == 2

    def test_step_preprocessamento_existe(self, pipeline_nao_treinado):
        """Primeiro step deve chamar-se 'preprocessamento'."""
        nomes = [nome for nome, _ in pipeline_nao_treinado.steps]
        assert "preprocessamento" in nomes

    def test_step_modelo_existe(self, pipeline_nao_treinado):
        """Segundo step deve chamar-se 'modelo'."""
        nomes = [nome for nome, _ in pipeline_nao_treinado.steps]
        assert "modelo" in nomes

    def test_modelo_eh_logistic_regression(self, pipeline_nao_treinado):
        """Modelo final deve ser LogisticRegression."""
        from sklearn.linear_model import LogisticRegression
        assert isinstance(pipeline_nao_treinado.named_steps["modelo"], LogisticRegression)

    def test_parametros_ridge(self, pipeline_nao_treinado):
        """LogisticRegression deve usar C=0.1 e class_weight='balanced' (Ridge L2)."""
        modelo = pipeline_nao_treinado.named_steps["modelo"]
        assert modelo.C == 0.1
        assert modelo.class_weight == "balanced"


# ---------------------------------------------------------------------------
# Testes — prever()
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_treinado():
    """
    Carrega o pipeline treinado do disco.
    Pula os testes se o arquivo não existir (modelo não gerado ainda).
    """
    from src.pipeline import MODEL_PATH, carregar_pipeline

    if not MODEL_PATH.exists():
        pytest.skip(
            "Pipeline não encontrado. Execute 'python src/pipeline.py' primeiro."
        )
    return carregar_pipeline()


@pytest.fixture
def cliente_exemplo():
    """Dicionário com dados de um cliente de exemplo (alto risco de churn)."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.70,
        "TotalCharges": 1028.40,
    }


class TestPrever:
    def test_retorna_dict(self, pipeline_treinado, cliente_exemplo):
        """prever() deve retornar um dicionário."""
        from api.main import _aplicar_encoding_entrada
        import pandas as pd

        df = _aplicar_encoding_entrada(pd.DataFrame([cliente_exemplo]))
        resultado = prever(df, pipeline=pipeline_treinado)
        assert isinstance(resultado, dict)

    def test_chaves_corretas(self, pipeline_treinado, cliente_exemplo):
        """Resultado deve conter 'churn_prediction' e 'churn_probability'."""
        from api.main import _aplicar_encoding_entrada
        import pandas as pd

        df = _aplicar_encoding_entrada(pd.DataFrame([cliente_exemplo]))
        resultado = prever(df, pipeline=pipeline_treinado)
        assert "churn_prediction" in resultado
        assert "churn_probability" in resultado

    def test_predicao_e_binaria(self, pipeline_treinado, cliente_exemplo):
        """churn_prediction deve ser 0 ou 1."""
        from api.main import _aplicar_encoding_entrada
        import pandas as pd

        df = _aplicar_encoding_entrada(pd.DataFrame([cliente_exemplo]))
        resultado = prever(df, pipeline=pipeline_treinado)
        assert resultado["churn_prediction"] in {0, 1}

    def test_probabilidade_entre_0_e_1(self, pipeline_treinado, cliente_exemplo):
        """churn_probability deve estar entre 0.0 e 1.0."""
        from api.main import _aplicar_encoding_entrada
        import pandas as pd

        df = _aplicar_encoding_entrada(pd.DataFrame([cliente_exemplo]))
        resultado = prever(df, pipeline=pipeline_treinado)
        assert 0.0 <= resultado["churn_probability"] <= 1.0