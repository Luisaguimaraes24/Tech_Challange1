"""
api/schemas.py
Modelos Pydantic para validação de entrada e saída da API FastAPI.

Campos espelham o dataset Telco Customer Churn após limpeza:
    - customerID removido (PII)
    - Churn não é enviado como input (é o que o modelo prediz)
    - Campos categóricos aceitos como strings (o pipeline trata internamente)
    - Campos numéricos com validação de range realista
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    """
    Dados de entrada para predição de churn de um cliente Telco.
    Todos os campos são obrigatórios e validados automaticamente pelo Pydantic.
    """

    # Demográficos
    gender: Literal["Male", "Female"] = Field(
        ..., description="Gênero do cliente"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Cliente é idoso? (0=Não, 1=Sim)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Cliente tem parceiro(a)?"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Cliente tem dependentes?"
    )

    # Conta
    tenure: int = Field(
        ..., ge=0, le=120, description="Meses como cliente (0–120)"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Possui serviço de telefone?"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Possui múltiplas linhas?"
    )

    # Internet
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de serviço de internet"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui segurança online?"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui backup online?"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui proteção de dispositivo?"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui suporte técnico?"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui streaming de TV?"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Possui streaming de filmes?"
    )

    # Contrato e pagamento
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Tipo de contrato"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Usa fatura sem papel?"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Método de pagamento")

    # Financeiro
    MonthlyCharges: float = Field(
        ..., ge=0.0, le=200.0, description="Cobranças mensais em R$ (0–200)"
    )
    TotalCharges: float = Field(
        ..., ge=0.0, description="Total cobrado desde o início do contrato"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredicaoOutput(BaseModel):
    """Resposta da predição de churn."""

    churn_prediction: Literal[0, 1] = Field(
        ..., description="Predição binária (1=Churn, 0=Não Churn)"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidade de churn (0.0–1.0)"
    )
    risco: Literal["Alto", "Médio", "Baixo"] = Field(
        ..., description="Classificação de risco baseada na probabilidade"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_prediction": 1,
                "churn_probability": 0.7842,
                "risco": "Alto",
            }
        }
    }


class HealthOutput(BaseModel):
    """Resposta do endpoint /health."""

    status: Literal["healthy", "unhealthy"]
    modelo_carregado: bool
    versao_api: str