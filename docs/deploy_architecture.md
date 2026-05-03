# Arquitetura de Deploy — Telco Churn Prediction

**Projeto:** Tech Challenge FIAP — Fase 1  
**Data:** Maio/2026  

---

## 1. Decisão Arquitetural: Real-Time vs Batch

### 1.1 Comparação das abordagens

| Critério | Batch (Agendado) | Real-Time (API) |
|---|---|---|
| **Latência** | Horas/dias (execução periódica) | Milissegundos (resposta imediata) |
| **Escalabilidade** | Alta (processa toda a base de uma vez) | Média (instâncias por demanda) |
| **Infraestrutura** | Job scheduler (cron, Airflow) | Servidor HTTP persistente |
| **Custo** | Baixo (recursos sob demanda) | Médio (instância sempre ativa) |
| **Atualização de predições** | Semanal/diária | Imediata por evento |
| **Integração com CRM** | Via arquivo/banco | Via HTTP REST |
| **Complexidade operacional** | Baixa | Média |
| **Casos de uso suportados** | Campanha semanal de retenção | Onboarding, atendimento, evento de risco |

### 1.2 Decisão adotada: Real-Time via API REST

**Escolha:** Deploy em tempo real com API FastAPI, com capacidade de uso tanto em modo batch (chamadas em lote via script) quanto em modo real-time (integração com CRM).

**Justificativa:**

1. **Flexibilidade de integração:** A API REST permite que qualquer sistema (CRM, call center, dashboard) consulte o risco de um cliente específico no momento da interação — sem esperar a próxima execução do job batch.

2. **Alinhamento com o SLO:** A meta de disponibilidade de 99% em horário comercial (definida no ML Canvas) é mais facilmente atendida com uma API persistente do que com um job agendado.

3. **Custo de FN:** O custo de um falso negativo (cliente que vai cancelar e não é identificado) é R$ 840 — 16,8× o custo de um FP. Qualquer evento que indique risco de churn (ligação ao suporte, aumento de fatura) deve poder acionar uma consulta imediata ao modelo.

4. **Evolução futura:** A arquitetura de API permite futura adição de streaming de eventos (Kafka, webhooks) sem redesenho — o modelo já está exposto como serviço.

5. **Compatibilidade com batch:** O uso batch semanal (lista de todos os clientes) pode ser implementado como um script que chama a API em loop ou que usa o pipeline diretamente — sem precisar de uma infraestrutura batch separada.

---

## 2. Arquitetura do Sistema

### 2.1 Diagrama de componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENTES DA API                          │
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │  Sistema CRM │   │ Script Batch │   │  Dashboard BI    │   │
│   │  (evento)    │   │  (semanal)   │   │  (consulta)      │   │
│   └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘   │
└──────────┼─────────────────┼────────────────────┼─────────────┘
           │                 │                    │
           │   HTTP POST /predict                 │
           └─────────────────┴────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI App   │
                    │   (port 8000)   │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │  Middleware  │ │  ← Logging estruturado
                    │ │  Latência   │ │  ← X-Process-Time-Ms
                    │ └─────────────┘ │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │  Pydantic   │ │  ← Validação de schema
                    │ │  Schemas    │ │  ← ClienteInput (19 campos)
                    │ └─────────────┘ │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │  sklearn    │ │  ← Pipeline (em memória)
                    │ │  Pipeline   │ │  ← Carregado no startup
                    │ └─────────────┘ │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  models/        │
                    │  churn_pipeline │  ← joblib serializado
                    │  .joblib        │
                    └─────────────────┘
```

### 2.2 Fluxo de uma requisição

```
1. Cliente envia POST /predict com JSON do perfil do cliente
2. Pydantic valida o schema (19 campos obrigatórios com tipos)
3. Middleware registra início do timer
4. Handler converte dict → DataFrame
5. _aplicar_encoding_entrada() faz encoding binário/ordinal
6. pipeline.predict_proba() produz probabilidade de churn
7. Classificação de risco: Alto/Médio/Baixo
8. Middleware registra latência no header X-Process-Time-Ms
9. Logger estruturado registra request + resultado em JSON
10. Resposta retorna: churn_prediction, churn_probability, risco
```

### 2.3 Endpoints disponíveis

| Endpoint | Método | Descrição | Response |
|---|---|---|---|
| `/health` | GET | Health check do serviço e do modelo | `{status, modelo_carregado, versao_api}` |
| `/predict` | POST | Predição de churn para um cliente | `{churn_prediction, churn_probability, risco}` |
| `/docs` | GET | Documentação interativa (Swagger UI) | Interface HTML |
| `/redoc` | GET | Documentação alternativa (ReDoc) | Interface HTML |

---

## 3. Stack Tecnológica

### 3.1 Componentes do serviço

| Componente | Tecnologia | Versão | Papel |
|---|---|---|---|
| **Framework web** | FastAPI | ≥ 0.111 | Servidor HTTP, routing, OpenAPI automático |
| **Servidor ASGI** | Uvicorn | ≥ 0.29 | Servidor de produção assíncrono |
| **Validação** | Pydantic v2 | ≥ 2.7 | Schema de entrada/saída, coerção de tipos |
| **Pipeline ML** | scikit-learn | ≥ 1.4 | Preprocessing + classificação |
| **Serialização** | joblib | — | Persistência do pipeline treinado |
| **Logging** | Python logging | — | Logs estruturados JSON sem prints |

### 3.2 Pipeline de ML (sklearn)

```
ColumnTransformer
├── StandardScaler → [tenure, MonthlyCharges, TotalCharges]
└── OneHotEncoder (drop='first') → [InternetService, Contract, PaymentMethod]
    └── remainder='passthrough' → demais features binárias/ordinais
        │
        └── LogisticRegression
            ├── penalty='l2' (Ridge)
            ├── C=0.1 (regularização forte)
            ├── solver='lbfgs'
            ├── class_weight='balanced'
            └── random_state=42
```

---

## 4. Modelo de Deployment

### 4.1 Ambiente local (desenvolvimento)

```bash
# Instalar dependências
make install         # pip install -e ".[dev]"

# Treinar o modelo
make train           # python src/pipeline.py

# Subir API com reload automático
make run             # uvicorn api.main:app --reload --port 8000
```

### 4.2 Ambiente de produção (recomendado)

```bash
# Sem reload, workers configuráveis
make run-prod        # uvicorn api.main:app --workers 1 --port 8000
```

Para produção com múltiplas instâncias, recomenda-se:

```bash
# Com Gunicorn como process manager
gunicorn api.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 2 \
    --bind 0.0.0.0:8000 \
    --timeout 30
```

### 4.3 Containerização (Docker — opcional)

Estrutura recomendada para containerização:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .

COPY src/ src/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--app-dir", "src", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**Build e execução:**
```bash
docker build -t telco-churn-api:1.0 .
docker run -p 8000:8000 telco-churn-api:1.0
```

### 4.4 Deploy em nuvem (opções)

| Plataforma | Estratégia | Complexidade |
|---|---|---|
| **Railway / Render** | Push do Dockerfile, variáveis de ambiente | Baixa — recomendada para MVP |
| **AWS App Runner** | Imagem Docker no ECR → App Runner | Média |
| **GCP Cloud Run** | Serverless container — escala para zero | Média — custo eficiente |
| **AWS ECS / Fargate** | Orquestração de containers gerenciada | Alta — produção enterprise |
| **Kubernetes** | Helm chart + Deployment + Service | Alta — máxima flexibilidade |

---

## 5. Considerações Operacionais

### 5.1 Gerenciamento de estado

O modelo é carregado **uma única vez no startup** da aplicação via `lifespan` context manager (FastAPI). Isso evita:

- Latência de carregamento por request
- Múltiplas desserializações do joblib em paralelo
- Inconsistência de estado entre requisições

O pipeline é armazenado em `_estado["pipeline"]` — uma variável global do módulo, thread-safe para operações de leitura (sklearn `predict` é stateless após o fit).

### 5.2 Latência esperada

| Operação | Latência estimada |
|---|---|
| Startup (carregamento do modelo) | 200–500ms (uma única vez) |
| Predição por request (P50) | < 10ms |
| Predição por request (P95) | < 50ms |
| Predição por request (P99) | < 100ms |

O modelo LogisticRegression com sparse input é extremamente rápido — o gargalo é o overhead HTTP e o encoding de entrada, não a inferência ML.

### 5.3 Tratamento de erros

| Código HTTP | Cenário |
|---|---|
| 200 | Predição realizada com sucesso |
| 422 | Schema inválido (Pydantic) — campo ausente, tipo incorreto |
| 503 | Modelo não carregado (startup falhou) |
| 500 | Erro interno não tratado |

### 5.4 Versionamento do modelo

A versão atual é definida em `VERSION = "1.0.0"` em `src/api/main.py` e retornada pelo endpoint `/health`. Para substituir o modelo em produção:

1. Treinar nova versão: `make train`
2. Substituir `models/churn_pipeline.joblib`
3. Atualizar `VERSION` no código
4. Reiniciar o serviço (ou usar blue-green deployment)
5. Registrar experimento no MLflow com as métricas da nova versão

---

## 6. Segurança

| Aspecto | Medida adotada |
|---|---|
| **PII no payload** | `customerID` não é aceito pelo schema Pydantic — nunca chega ao modelo |
| **Injeção de input** | Pydantic v2 com tipos estritos — sem eval, sem execução dinâmica |
| **Autenticação** | Não implementada na v1 — recomendada para produção (API Key / OAuth2) |
| **HTTPS** | Responsabilidade do proxy reverso (nginx, Cloudflare) em produção |
| **Rate limiting** | Não implementado — recomendado para produção (middleware ou API Gateway) |
| **Logging** | Logs em JSON sem dados pessoais (tenure, Contract, InternetService apenas) |

---

*Documento elaborado como entregável formal da Etapa 4 — Tech Challenge FIAP Fase 1.*
