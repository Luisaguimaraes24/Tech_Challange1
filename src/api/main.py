"""
api/main.py
API de inferência de churn — Telco Customer Churn (FIAP Tech Challenge Fase 1)

Endpoints:
    GET  /health   → status da API e do modelo
    POST /predict  → predição de churn com probabilidade e classificação de risco

Recursos:
    - Validação automática via Pydantic (schemas.py)
    - Logging estruturado em JSON (sem prints)
    - Middleware de latência: registra tempo de resposta de cada requisição
    - Modelo carregado uma única vez no startup (lifespan)
    - Documentação automática: /docs (Swagger) e /redoc

Uso:
    uvicorn api.main:app --reload
    # ou via Makefile: make run
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import ClienteInput, HealthOutput, PredicaoOutput
from src.data.preprocessing import limpar_dados
from src.pipeline import carregar_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Estado global — modelo carregado uma única vez
# ---------------------------------------------------------------------------
_estado = {"pipeline": None}

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Lifespan — carrega o modelo no startup, libera no shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação: carrega modelo no startup."""
    logger.info("Iniciando API — carregando pipeline...")
    try:
        _estado["pipeline"] = carregar_pipeline()
        logger.info("Pipeline carregado com sucesso.")
    except FileNotFoundError as exc:
        logger.error("Falha ao carregar pipeline: %s", exc)
        logger.error(
            "Execute 'python src/pipeline.py' para treinar e salvar o modelo antes de subir a API."
        )
    yield
    logger.info("API encerrada.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telco Churn Prediction API",
    description=(
        "API de inferência para predição de churn de clientes de telecomunicações. "
        "Modelo: LogisticRegression + Ridge L2 (melhor modelo da Etapa 2 — AUC-ROC 0.841)."
    ),
    version=VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware de latência
# ---------------------------------------------------------------------------

@app.middleware("http")
async def middleware_latencia(request: Request, call_next):
    """
    Middleware que mede e registra o tempo de resposta de cada requisição.
    Adiciona o header X-Process-Time-Ms na resposta.
    """
    inicio = time.perf_counter()
    response = await call_next(request)
    duracao_ms = round((time.perf_counter() - inicio) * 1000, 2)

    response.headers["X-Process-Time-Ms"] = str(duracao_ms)

    logger.info(
        "method=%s path=%s status=%d latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duracao_ms,
    )

    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthOutput, tags=["Monitoramento"])
def health():
    """
    Verifica o status da API e se o modelo está carregado.
    Usado para health checks em produção (Docker, Kubernetes, etc).
    """
    modelo_ok = _estado["pipeline"] is not None

    logger.info("Health check — modelo_carregado=%s", modelo_ok)

    return HealthOutput(
        status="healthy" if modelo_ok else "unhealthy",
        modelo_carregado=modelo_ok,
        versao_api=VERSION,
    )


@app.post("/predict", response_model=PredicaoOutput, tags=["Predição"])
def predict(cliente: ClienteInput):
    """
    Realiza a predição de churn para um cliente.

    Recebe os dados do cliente como JSON, aplica o pipeline de preprocessing
    e retorna a predição binária, a probabilidade e a classificação de risco.

    - **churn_prediction**: 1 = alto risco de churn | 0 = baixo risco
    - **churn_probability**: probabilidade de churn (0.0 a 1.0)
    - **risco**: Alto (≥ 0.6) | Médio (0.4–0.6) | Baixo (< 0.4)
    """
    if _estado["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Tente novamente em instantes ou contate o suporte.",
        )

    # Converter Pydantic → dict → DataFrame
    dados_dict = cliente.model_dump()
    logger.info(
        "Predição solicitada — tenure=%s contract=%s internet=%s",
        dados_dict.get("tenure"),
        dados_dict.get("Contract"),
        dados_dict.get("InternetService"),
    )

    # Aplicar limpeza (encoding binário/ordinal, igual ao treinamento)
    import pandas as pd

    df_raw = pd.DataFrame([dados_dict])
    # O pipeline já tem o preprocessador, mas precisamos aplicar
    # o encoding de preprocessing.py que ocorre antes do sklearn Pipeline.
    # Aqui simulamos o que limpar_dados() faz nas colunas binárias/ordinais,
    # sem precisar do CSV completo (já recebemos os dados limpos do cliente).
    df_input = _aplicar_encoding_entrada(df_raw)

    # Predição
    pipeline = _estado["pipeline"]
    predicao = int(pipeline.predict(df_input)[0])
    probabilidade = float(pipeline.predict_proba(df_input)[0][1])

    # Classificação de risco
    if probabilidade >= 0.6:
        risco = "Alto"
    elif probabilidade >= 0.4:
        risco = "Médio"
    else:
        risco = "Baixo"

    logger.info(
        "Resultado — churn=%d probabilidade=%.4f risco=%s",
        predicao, probabilidade, risco,
    )

    return PredicaoOutput(
        churn_prediction=predicao,
        churn_probability=round(probabilidade, 4),
        risco=risco,
    )


# ---------------------------------------------------------------------------
# Helper — encoding de entrada (espelha preprocessing.py)
# ---------------------------------------------------------------------------

def _aplicar_encoding_entrada(df):
    """
    Aplica o mesmo encoding que preprocessing.limpar_dados() faz,
    mas apenas nas colunas recebidas pela API (sem target, sem customerID).
    """
    import pandas as pd

    df = df.copy()

    # Gender
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    # Binárias Yes/No → 0/1
    binarias = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binarias:
        if col in df.columns:
            df[col] = df[col].map({"No": 0, "Yes": 1})

    # Serviços → 0/1/2
    _mapa_servico = {
        "No phone service": 0,
        "No internet service": 0,
        "No": 1,
        "Yes": 2,
    }
    servicos = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in servicos:
        if col in df.columns:
            df[col] = df[col].map(_mapa_servico)

    # SeniorCitizen já é 0/1
    # InternetService, Contract, PaymentMethod → OneHotEncoder no Pipeline

    return df


# ---------------------------------------------------------------------------
# Handler de erros genérico
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def handler_erro_generico(request: Request, exc: Exception):
    logger.error("Erro não tratado em %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erro interno do servidor. Consulte os logs para mais detalhes."},
    )