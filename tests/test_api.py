"""
tests/test_api.py
Testes da API FastAPI — smoke test, /health, /predict

Usa httpx.TestClient para testar os endpoints sem subir o servidor.

Referência: Eng. Software, Aula 03 — Testes Automatizados
             APIs, Aulas 01–04
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixture: cliente de teste com modelo mockado
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cliente_mock():
    """
    TestClient da API com o pipeline mockado.
    Não depende do arquivo .joblib — funciona mesmo sem treino prévio.
    """
    # Mock do pipeline sklearn
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [1]
    mock_pipeline.predict_proba.return_value = [[0.2158, 0.7842]]

    import api.main as main_module

    main_module._estado["pipeline"] = mock_pipeline
    client = TestClient(main_module.app, raise_server_exceptions=True)
    return client


@pytest.fixture
def payload_valido():
    """Payload JSON válido para o endpoint /predict."""
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


# ---------------------------------------------------------------------------
# Smoke tests — a API sobe sem crashar
# ---------------------------------------------------------------------------

class TestSmoke:
    def test_api_responde(self, cliente_mock):
        """Smoke test: a API deve responder a qualquer requisição sem crashar."""
        response = cliente_mock.get("/health")
        assert response.status_code in {200, 503}

    def test_docs_acessivel(self, cliente_mock):
        """Swagger UI deve estar disponível em /docs."""
        response = cliente_mock.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self, cliente_mock):
        """Schema OpenAPI deve estar disponível em /openapi.json."""
        response = cliente_mock.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]


# ---------------------------------------------------------------------------
# Testes — GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_200(self, cliente_mock):
        """GET /health deve retornar 200."""
        response = cliente_mock.get("/health")
        assert response.status_code == 200

    def test_health_campos(self, cliente_mock):
        """Resposta deve conter status, modelo_carregado e versao_api."""
        response = cliente_mock.get("/health")
        body = response.json()
        assert "status" in body
        assert "modelo_carregado" in body
        assert "versao_api" in body

    def test_health_modelo_carregado(self, cliente_mock):
        """Com o modelo mockado, modelo_carregado deve ser True."""
        response = cliente_mock.get("/health")
        body = response.json()
        assert body["modelo_carregado"] is True
        assert body["status"] == "healthy"

    def test_health_header_latencia(self, cliente_mock):
        """Middleware de latência deve adicionar o header X-Process-Time-Ms."""
        response = cliente_mock.get("/health")
        assert "x-process-time-ms" in response.headers


# ---------------------------------------------------------------------------
# Testes — POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_200(self, cliente_mock, payload_valido):
        """POST /predict com payload válido deve retornar 200."""
        response = cliente_mock.post("/predict", json=payload_valido)
        assert response.status_code == 200

    def test_predict_campos_resposta(self, cliente_mock, payload_valido):
        """Resposta deve conter churn_prediction, churn_probability e risco."""
        response = cliente_mock.post("/predict", json=payload_valido)
        body = response.json()
        assert "churn_prediction" in body
        assert "churn_probability" in body
        assert "risco" in body

    def test_predict_predicao_binaria(self, cliente_mock, payload_valido):
        """churn_prediction deve ser 0 ou 1."""
        response = cliente_mock.post("/predict", json=payload_valido)
        body = response.json()
        assert body["churn_prediction"] in {0, 1}

    def test_predict_probabilidade_range(self, cliente_mock, payload_valido):
        """churn_probability deve estar entre 0.0 e 1.0."""
        response = cliente_mock.post("/predict", json=payload_valido)
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_predict_risco_valido(self, cliente_mock, payload_valido):
        """risco deve ser 'Alto', 'Médio' ou 'Baixo'."""
        response = cliente_mock.post("/predict", json=payload_valido)
        body = response.json()
        assert body["risco"] in {"Alto", "Médio", "Baixo"}

    def test_predict_payload_incompleto_422(self, cliente_mock):
        """Payload incompleto deve retornar 422 (Unprocessable Entity)."""
        response = cliente_mock.post("/predict", json={"tenure": 12})
        assert response.status_code == 422

    def test_predict_campo_invalido_422(self, cliente_mock, payload_valido):
        """Valor inválido em campo Literal deve retornar 422."""
        payload_invalido = payload_valido.copy()
        payload_invalido["Contract"] = "Contrato Inválido"
        response = cliente_mock.post("/predict", json=payload_invalido)
        assert response.status_code == 422

    def test_predict_tenure_negativo_422(self, cliente_mock, payload_valido):
        """Tenure negativo deve retornar 422 (ge=0 no schema)."""
        payload_invalido = payload_valido.copy()
        payload_invalido["tenure"] = -1
        response = cliente_mock.post("/predict", json=payload_invalido)
        assert response.status_code == 422

    def test_predict_header_latencia(self, cliente_mock, payload_valido):
        """Middleware de latência deve estar presente no /predict também."""
        response = cliente_mock.post("/predict", json=payload_valido)
        assert "x-process-time-ms" in response.headers

    def test_predict_sem_modelo_503(self):
        """Sem modelo carregado, /predict deve retornar 503."""
        import api.main as main_module

        pipeline_original = main_module._estado["pipeline"]
        main_module._estado["pipeline"] = None

        client = TestClient(main_module.app, raise_server_exceptions=False)
        payload = {
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
        response = client.post("/predict", json=payload)
        assert response.status_code == 503

        # Restaura o mock para não afetar outros testes
        main_module._estado["pipeline"] = pipeline_original