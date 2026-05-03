# Telco Customer Churn Prediction

Previsão de cancelamento de clientes em telecomunicações usando Machine Learning.  
**Tech Challenge FIAP — Fase 1 (Machine Learning Engineering)**

---

## Objetivo

Identificar clientes com alto risco de cancelamento (*churn*) para viabilizar ações preventivas de retenção. O modelo retorna probabilidade de churn (0–1) e classificação de risco (Alto/Médio/Baixo) via API REST em tempo real.

**Resultado:** LogisticRegression + Ridge L2 — AUC-ROC 0.841 | Recall 0.778 | Resultado líquido R$ 230.340 no conjunto de teste (1.409 clientes).

---

## Índice

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Treinamento](#treinamento)
- [API de Inferência](#api-de-inferência)
- [Testes](#testes)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura](#arquitetura)
- [Documentação](#documentação)

---

## Pré-requisitos

- Python ≥ 3.10
- pip ≥ 23
- (Opcional) Docker para containerização

---

## Instalação

```bash
# 1. Clonar o repositório
git clone <url-do-repositório>
cd Tech_Challange1

# 2. Instalar dependências (produção + dev)
make install
# equivalente a: pip install -e ".[dev]"
```

---

## Treinamento

O dataset bruto já está em `data/raw/Telco-Customer-Churn.csv`.

```bash
# Treinar o pipeline e salvar em models/churn_pipeline.joblib
make train
# equivalente a: python src/pipeline.py
```

O pipeline serializado será gerado em `models/churn_pipeline.joblib`.  
Todos os experimentos são registrados automaticamente no MLflow (diretório `mlruns/`).

```bash
# Visualizar experimentos no MLflow UI
mlflow ui
# Acesse: http://localhost:5000
```

---

## API de Inferência

### Subir o servidor

```bash
# Desenvolvimento (com reload automático)
make run
# equivalente a: uvicorn api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

# Produção (sem reload)
make run-prod
```

A API estará disponível em `http://localhost:8000`.  
Documentação interativa: `http://localhost:8000/docs`

### Verificar status (health check)

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "versao_api": "1.0.0"
}
```

### Realizar uma predição

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 151.65
  }'
```

**Resposta:**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7834,
  "risco": "Alto"
}
```

### Interpretação da resposta

| Campo | Tipo | Descrição |
|---|---|---|
| `churn_prediction` | int (0 ou 1) | 1 = cliente em risco de churn |
| `churn_probability` | float (0.0–1.0) | Probabilidade de churn |
| `risco` | string | `Alto` (≥0.6) · `Médio` (0.4–0.6) · `Baixo` (<0.4) |

### Campos obrigatórios da requisição

| Campo | Tipo | Valores aceitos |
|---|---|---|
| `gender` | string | `"Male"`, `"Female"` |
| `SeniorCitizen` | int | `0`, `1` |
| `Partner` | string | `"Yes"`, `"No"` |
| `Dependents` | string | `"Yes"`, `"No"` |
| `tenure` | int | 0–72 (meses) |
| `PhoneService` | string | `"Yes"`, `"No"` |
| `MultipleLines` | string | `"Yes"`, `"No"`, `"No phone service"` |
| `InternetService` | string | `"DSL"`, `"Fiber optic"`, `"No"` |
| `OnlineSecurity` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `OnlineBackup` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `DeviceProtection` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `TechSupport` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `StreamingTV` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `StreamingMovies` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `Contract` | string | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `PaperlessBilling` | string | `"Yes"`, `"No"` |
| `PaymentMethod` | string | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |
| `MonthlyCharges` | float | Valor da fatura mensal |
| `TotalCharges` | float | Total gasto pelo cliente |

---

## Testes

```bash
# Rodar todos os testes
make test

# Com relatório de cobertura
make test-cov

# Por suíte específica
make test-schema     # validação de schema (pandera)
make test-pipeline   # testes unitários do pipeline sklearn
make test-api        # testes de integração da API
```

### Linting

```bash
make lint            # verificar sem corrigir
make lint-fix        # verificar e corrigir automaticamente
```

---

## Estrutura do Projeto

```
Tech_Challange1/
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv    # Dataset bruto (IBM)
│   └── processed/
│       └── telco_clean.csv             # Dataset após limpeza
│
├── docs/
│   ├── ML_Canvas.md                    # Canvas do projeto (CRISP-DM)
│   ├── etapa2_entregavel.md            # Comparação de 6 modelos
│   ├── model_card.md                   # Model Card (Etapa 4)
│   ├── deploy_architecture.md          # Arquitetura de deploy (Etapa 4)
│   └── monitoring_plan.md              # Plano de monitoramento (Etapa 4)
│
├── models/
│   ├── churn_pipeline.joblib           # Pipeline sklearn serializado
│   └── mlp_best.pt                     # Checkpoint da MLP PyTorch
│
├── notebooks/
│   ├── 01_eda.py                       # Análise exploratória
│   ├── 02_baselines.py                 # DummyClassifier + LogisticRegression
│   ├── 03_ensemble.py                  # Random Forest + Gradient Boosting
│   └── 04_mlp.py                       # MLP PyTorch com early stopping
│
├── src/
│   ├── api/
│   │   ├── main.py                     # FastAPI app (endpoints + middleware)
│   │   └── schemas.py                  # Schemas Pydantic v2
│   ├── data/
│   │   └── preprocessing.py            # Limpeza e encoding dos dados
│   ├── features/
│   │   └── build_features.py           # ColumnTransformer + split estratificado
│   ├── models/
│   │   ├── mlp.py                      # Arquitetura ChurnMLP (PyTorch)
│   │   └── trainer.py                  # Loop de treino com early stopping
│   ├── utils/
│   │   └── logger.py                   # Logger estruturado JSON
│   └── pipeline.py                     # Pipeline de produção (treino + inferência)
│
├── tests/
│   ├── test_api.py                     # Testes de integração da API
│   ├── test_pipeline.py                # Testes unitários do pipeline
│   └── test_schema.py                  # Validação de schema (pandera)
│
├── outputs/                            # Gráficos e visualizações geradas
├── mlruns/                             # Experimentos MLflow (local)
├── Makefile                            # Targets de automação
├── pyproject.toml                      # Dependências e configuração
└── README.md
```

---

## Arquitetura

### Modelo de produção

```
LogisticRegression + Ridge L2 (C=0.1, class_weight=balanced)
    ↑
ColumnTransformer
├── StandardScaler → [tenure, MonthlyCharges, TotalCharges]
└── OneHotEncoder  → [InternetService, Contract, PaymentMethod]
    └── passthrough → demais features (binárias/ordinais)
```

### Fluxo de inferência

```
POST /predict
    → Pydantic valida schema (19 campos)
    → Encoding de entrada (binárias + ordinais)
    → pipeline.predict_proba() → probabilidade de churn
    → Classificação de risco (Alto/Médio/Baixo)
    → Resposta JSON + log estruturado
```

### Desempenho no conjunto de teste (1.409 clientes)

| Métrica | Resultado | Meta |
|---|---|---|
| AUC-ROC | 0.841 | ≥ 0.80 ✅ |
| Recall | 0.778 | ≥ 0.70 ✅ |
| F1-Score | 0.615 | ≥ 0.68 ⚠️ |
| Resultado líquido | R$ 230.340 | — |
| Overfitting gap | 0.015 | < 0.05 ✅ |

### Por que LogisticRegression foi escolhida?

Dentre os 6 modelos avaliados (DummyClassifier, LogReg baseline, **LogReg Ridge L2**, Random Forest, Gradient Boosting, MLP PyTorch):

- **Melhor resultado líquido de negócio** (R$ 230.340) — supera MLP e Random Forest
- **Recall 0.778** — atinge a meta de ≥ 0.70 (detecta 78% dos churners reais)
- **Zero overfitting** (gap treino/teste = 0.015)
- **Interpretável** — coeficientes explicáveis ao time de negócio
- **Rápido** — inferência em milissegundos, sem GPU necessária

---

## Documentação

| Documento | Descrição |
|---|---|
| [docs/ML_Canvas.md](docs/ML_Canvas.md) | Objetivo de negócio, métricas técnicas, stakeholders, SLOs |
| [docs/etapa2_entregavel.md](docs/etapa2_entregavel.md) | Comparação de 6 modelos com análise financeira |
| [docs/model_card.md](docs/model_card.md) | Performance, limitações, vieses, cenários de falha |
| [docs/deploy_architecture.md](docs/deploy_architecture.md) | Justificativa da arquitetura real-time, stack, containerização |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | Métricas, alertas, playbook de resposta, estratégia de retreino |

---

## Pipeline completo (do zero)

```bash
# Instalar + treinar + lint + testar
make all
```

---

*Projeto desenvolvido como Tech Challenge FIAP Fase 1 — Machine Learning Engineering.*
