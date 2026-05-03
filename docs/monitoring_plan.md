# Plano de Monitoramento — Telco Churn Prediction

**Projeto:** Tech Challenge FIAP — Fase 1  
**Versão:** 1.0  
**Data:** Maio/2026  

---

## 1. Objetivos do Monitoramento

O monitoramento de um modelo em produção tem três propósitos:

1. **Detectar degradação de desempenho** antes que impacte o negócio
2. **Identificar data drift** — mudanças na distribuição dos dados de entrada
3. **Garantir disponibilidade e latência** adequadas do serviço de inferência

O modelo Telco Churn Classifier v1.0 opera em contexto de negócio de alto custo de falso negativo (FN = R$ 840 por churner não detectado), tornando a detecção precoce de degradação crítica para o ROI da operação.

---

## 2. Camadas de Monitoramento

```
┌─────────────────────────────────────────────────────────────┐
│  Camada 1: Infraestrutura                                    │
│  Disponibilidade · Latência · Throughput · Erros HTTP        │
├─────────────────────────────────────────────────────────────┤
│  Camada 2: Dados de entrada (Data Drift)                     │
│  Distribuição de features · PSI · KS Test · Missing rate     │
├─────────────────────────────────────────────────────────────┤
│  Camada 3: Output do modelo (Prediction Drift)               │
│  Distribuição de scores · Taxa de churn predita · Skew       │
├─────────────────────────────────────────────────────────────┤
│  Camada 4: Desempenho do modelo (Model Performance)          │
│  AUC-ROC · Recall · Precisão · F1 · Resultado líquido        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Métricas por Camada

### 3.1 Camada 1 — Infraestrutura

| Métrica | Descrição | Coleta |
|---|---|---|
| **Disponibilidade** | % do tempo que `/health` retorna 200 | Probe HTTP a cada 1 min |
| **Latência P50** | Mediana do tempo de resposta do `/predict` | Middleware `X-Process-Time-Ms` |
| **Latência P95** | Percentil 95 do tempo de resposta | Middleware `X-Process-Time-Ms` |
| **Latência P99** | Percentil 99 do tempo de resposta | Middleware `X-Process-Time-Ms` |
| **Taxa de erro 5xx** | % de respostas com status ≥ 500 | Logs estruturados |
| **Taxa de erro 4xx** | % de respostas com status 422 (schema inválido) | Logs estruturados |
| **Throughput (RPS)** | Requisições por segundo ao `/predict` | Logs estruturados |
| **Modelo carregado** | `modelo_carregado` do `/health` | Probe `/health` |

**SLOs de referência:**

| SLO | Meta |
|---|---|
| Disponibilidade | ≥ 99% em horário comercial (9h–18h) |
| Latência P95 | ≤ 200ms |
| Latência P99 | ≤ 500ms |
| Taxa de erro 5xx | ≤ 0.1% |

### 3.2 Camada 2 — Data Drift (Distribuição de Features)

O **PSI (Population Stability Index)** compara a distribuição de cada feature entre o dataset de treino (referência) e os dados que chegam em produção.

**Interpretação do PSI:**

| PSI | Interpretação | Ação |
|---|---|---|
| < 0.10 | Distribuição estável | Nenhuma |
| 0.10 – 0.20 | Drift moderado | Investigar |
| > 0.20 | Drift severo | Alerta + retreinamento planejado |

**Features monitoradas por PSI (mensal):**

| Feature | Tipo | Referência de treino |
|---|---|---|
| `tenure` | Numérica | Média ≈ 32 meses, DP ≈ 24 |
| `MonthlyCharges` | Numérica | Média ≈ R$ 65, DP ≈ R$ 30 |
| `TotalCharges` | Numérica | Média ≈ R$ 2.280, DP ≈ R$ 2.267 |
| `Contract` | Categórica | Month-to-month: 55%, One year: 21%, Two year: 24% |
| `InternetService` | Categórica | DSL: 34%, Fiber optic: 44%, No: 22% |
| `PaymentMethod` | Categórica | Electronic check: 33%, outros: ~22% cada |
| `SeniorCitizen` | Binária | 16% idosos |

**Métricas complementares:**

| Métrica | Descrição |
|---|---|
| **KS Test (p-valor)** | Teste de Kolmogorov-Smirnov para features numéricas — detecta mudança de distribuição |
| **Missing rate por feature** | % de campos nulos ou com valor inválido por feature |
| **Taxa de novos valores categóricos** | % de valores OHE não vistos no treino (handle_unknown="ignore" no pipeline) |

### 3.3 Camada 3 — Prediction Drift (Output do Modelo)

| Métrica | Descrição | Frequência |
|---|---|---|
| **Score médio de churn** | Média de `churn_probability` nas últimas N predições | Diária |
| **Taxa de classificação Alto Risco** | % de clientes com score ≥ 0.6 | Diária |
| **Distribuição de risco** | % em cada faixa (Alto/Médio/Baixo) | Semanal |
| **PSI do score** | Comparação da distribuição do score vs. distribuição no treino | Mensal |

**Sinal de alerta:** Se a taxa de classificação "Alto Risco" mudar mais de 10 pontos percentuais em relação à baseline de treino (~26% churn histórico), investigar causa raiz antes de agir.

### 3.4 Camada 4 — Desempenho do Modelo

Esta é a camada mais direta — requer **dados rotulados em produção** (confirmação de churn real após N dias).

| Métrica | Baseline (treino) | Alerta | Crítico |
|---|---|---|---|
| **AUC-ROC** | 0.841 | < 0.80 (−5pp) | < 0.75 (−10pp) |
| **Recall** | 0.778 | < 0.70 (−8pp) | < 0.60 (−18pp) |
| **Precisão** | 0.508 | < 0.40 (−11pp) | < 0.30 (−21pp) |
| **F1-Score** | 0.615 | < 0.55 (−10pp) | < 0.45 (−17pp) |
| **Resultado líquido mensal** | R$ 230.340 (teste) | Queda > 20% | Queda > 40% |

**Desafio:** Em produção, o label verdadeiro (se o cliente efetivamente cancelou) só fica disponível 30–90 dias após a predição. Estratégias:

1. **Janela deslizante:** Avaliar métricas mensalmente com churns confirmados do período anterior
2. **Shadow mode:** Manter o modelo anterior rodando em paralelo por 30 dias após uma atualização, comparando outputs
3. **Proxy de degradação:** Usar PSI e prediction drift como proxies antes dos labels chegarem

---

## 4. Fontes de Dados para Monitoramento

| Fonte | Dados disponíveis | Implementação atual |
|---|---|---|
| **Logs da API** | Latência, método, path, status, tenure, contract, internet | `src/utils/logger.py` — JSON estruturado |
| **Endpoint `/health`** | Status do serviço, modelo carregado, versão | Implementado em `src/api/main.py` |
| **Header de resposta** | `X-Process-Time-Ms` por request | Middleware implementado |
| **MLflow** | Métricas de treino e validação, parâmetros | `mlruns/` local |
| **Dados de produção** | Features de entrada (sem PII) | Logar no payload do predict |

### 4.1 Formato dos logs estruturados (JSON)

Cada predição gera dois registros de log:

```json
// Log de request (início)
{
  "level": "INFO",
  "module": "api.main",
  "message": "Predição solicitada",
  "tenure": 12,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic"
}

// Log de resposta (fim)
{
  "level": "INFO",
  "module": "api.main",
  "message": "Resultado",
  "churn": 1,
  "probabilidade": 0.7234,
  "risco": "Alto",
  "latency_ms": 8.45,
  "status": 200
}
```

---

## 5. Alertas e Thresholds

### 5.1 Matriz de alertas

| Alerta | Condição | Severidade | Canal | Resposta esperada |
|---|---|---|---|---|
| API Indisponível | `/health` retorna != 200 por > 2 min | 🔴 Crítico | PagerDuty / SMS | Investigar imediatamente → reiniciar serviço |
| Latência Alta | P99 > 1000ms por > 5 min | 🟠 Alto | Slack #alertas-ml | Verificar carga, memória, modelo |
| AUC-ROC Degradado | AUC < 0.80 (queda ≥ 5pp) | 🟠 Alto | Slack #alertas-ml | Análise de drift → planejar retreino |
| Recall Degradado | Recall < 0.70 (queda ≥ 8pp) | 🟠 Alto | Slack #alertas-ml | Análise de drift → retreino urgente |
| PSI Crítico | PSI > 0.20 em qualquer feature relevante | 🟡 Médio | Email semanal | Análise de causa raiz, planejar retreino |
| PSI Moderado | PSI 0.10–0.20 em feature relevante | 🔵 Info | Dashboard | Investigar + monitorar mais de perto |
| Score Drift | Taxa de Alto Risco muda > 10pp | 🟡 Médio | Dashboard | Revisar distribuição de entrada |
| Taxa de Erro 422 | > 5% das requests com 422 | 🟡 Médio | Log + Email | Verificar sistema de origem dos dados |
| Modelo não carregado | `modelo_carregado: false` no `/health` | 🔴 Crítico | PagerDuty | Verificar `models/churn_pipeline.joblib` |

### 5.2 Escalada de alertas

```
Nível 1 (Automático): Log + métrica no dashboard
    ↓ sem ação em 15 min
Nível 2 (Notificação): Slack / email para equipe de ML
    ↓ sem ação em 1h
Nível 3 (Escalada): PagerDuty / SMS para engenheiro de plantão
    ↓ sem ação em 4h (durante horário comercial)
Nível 4 (Gestor): Notificação ao tech lead + plano de contingência
```

---

## 6. Playbook de Resposta a Incidentes

### 6.1 API Indisponível (503 / Health check falha)

**Diagnóstico:**
```bash
curl http://localhost:8000/health
# Se retornar 503 ou timeout:

# 1. Verificar processo
ps aux | grep uvicorn

# 2. Verificar logs
tail -f logs/api.log

# 3. Verificar modelo
ls -la models/churn_pipeline.joblib
```

**Ações:**
1. Se o processo morreu → reiniciar com `make run-prod`
2. Se o modelo não existe → executar `make train` e reiniciar
3. Se erro de importação → verificar `make install` e dependências

### 6.2 Degradação de AUC-ROC (< 0.80)

**Diagnóstico:**
1. Calcular PSI das features de entrada do período
2. Identificar features com PSI > 0.10
3. Verificar se houve mudança no negócio (novos planos, política de preços)

**Ações por causa raiz:**

| Causa | Ação |
|---|---|
| Data drift em features de contrato | Retreinar com dados mais recentes (últimos 3–6 meses) |
| Mudança no perfil demográfico | Coletar novos dados representativos + retreinar |
| Bug no sistema de origem | Corrigir pipeline de dados + investigar predições afetadas |
| Seasonal pattern | Treinar com dados do mesmo período do ano anterior |

### 6.3 PSI > 0.20 em Feature Relevante

**Diagnóstico:**
```python
# Calcular PSI para a feature afetada
from scipy.stats import ks_2samp
ks_stat, p_valor = ks_2samp(dados_treino['tenure'], dados_producao['tenure'])
# p_valor < 0.05 → distribuições significativamente diferentes
```

**Ações:**
1. **Imediato:** Verificar se é problema de dados (bug de ingestão) ou mudança real
2. **Curto prazo:** Se mudança real confirmada → agendar retreinamento
3. **Comunicação:** Notificar equipe de negócio sobre possível redução de precisão

### 6.4 Recall < 0.70 (muitos churners não detectados)

**Diagnóstico:**
1. Verificar distribuição do score — o modelo está concentrando predições em Baixo Risco?
2. Verificar se houve mudança no tipo de clientes que estão cancelando
3. Calcular confusion matrix com dados rotulados do período

**Ações:**
1. **Ajuste de threshold:** Diminuir threshold de 0.5 para 0.35–0.40 para aumentar Recall imediatamente (sem retreinamento)
2. **Retreinamento:** Se drift confirmado → retreinar com dados recentes
3. **Comunicação:** Alertar equipe de retenção que o modelo pode estar perdendo churners até normalização

---

## 7. Estratégia de Retreinamento

### 7.1 Gatilhos de retreinamento

| Gatilho | Tipo | Frequência de verificação |
|---|---|---|
| **Agendado** | Trimestral (proativo) | Automático — a cada 3 meses |
| **PSI > 0.20** | Drift de dados | Mensal |
| **AUC-ROC < 0.80** | Degradação de performance | Contínuo (com dados rotulados) |
| **Recall < 0.70** | Degradação crítica de negócio | Contínuo (com dados rotulados) |
| **Mudança de negócio** | Novos planos, fusão, mudança tarifária | Ad-hoc (comunicado pelo time de negócio) |

### 7.2 Processo de retreinamento

```
1. Coletar dados rotulados do período (confirmar churns reais)
2. Unir com dados históricos de treino (janela deslizante ou full retrain)
3. Executar EDA incremental — verificar se novas features são necessárias
4. Treinar com mesmo pipeline (make train) ou ajustar parâmetros
5. Avaliar métricas no conjunto de teste do novo período
6. Comparar com modelo atual — deve superar em ≥ 1pp em AUC-ROC ou Recall
7. Registrar no MLflow com tag 'v2.0' (ou versão correspondente)
8. Deploy com blue-green: manter v1.0 ativa até v2.0 validada por 72h
9. Atualizar Model Card com novas métricas
```

### 7.3 Critério de aceitação para novo modelo

O novo modelo é aceito se:
- AUC-ROC ≥ 0.80 **E**
- Recall ≥ 0.70 **E**
- Resultado líquido ≥ ao modelo atual no conjunto de teste recente

---

## 8. Infraestrutura de Monitoramento Recomendada

| Componente | Ferramenta Open Source | Ferramenta Gerenciada |
|---|---|---|
| **Coleta de métricas** | Prometheus | Datadog, New Relic |
| **Visualização** | Grafana | Grafana Cloud, DataDog |
| **Alertas** | Alertmanager | PagerDuty, OpsGenie |
| **Logs** | ELK Stack (Elasticsearch + Kibana) | Datadog Logs, AWS CloudWatch |
| **ML Monitoring** | Evidently AI | WhyLabs, Arize AI |
| **Experiment Tracking** | MLflow (já implementado) | MLflow no Databricks |

### 8.1 Dashboard mínimo (v1 — sem infraestrutura adicional)

Usando apenas os logs estruturados já implementados, um script Python pode gerar relatório semanal com:

```python
# Exemplo de métricas deriváveis dos logs existentes
import json
import pandas as pd

logs = pd.read_json('logs/api.log', lines=True)

# Latência
print(f"P50: {logs['latency_ms'].quantile(0.50):.1f}ms")
print(f"P95: {logs['latency_ms'].quantile(0.95):.1f}ms")
print(f"P99: {logs['latency_ms'].quantile(0.99):.1f}ms")

# Distribuição de risco
print(logs['risco'].value_counts(normalize=True))

# Taxa de churn predito
print(f"Churn rate: {logs['churn'].mean():.1%}")
```

---

## 9. Calendário de Monitoramento

| Frequência | Atividade | Responsável |
|---|---|---|
| **Contínuo** | Health check automático do `/health` | Infraestrutura |
| **Diário** | Verificar latência P95/P99 e erros 5xx | Eng. de dados |
| **Semanal** | Analisar distribuição de scores e taxa de Alto Risco | Cientista de dados |
| **Mensal** | Calcular PSI por feature; relatório de monitoramento | Cientista de dados |
| **Trimestral** | Coletar dados rotulados; avaliar AUC-ROC e Recall; decidir sobre retreinamento | Equipe ML + Negócio |
| **Semestral** | Revisão completa do Model Card; auditoria de fairness | Tech Lead |

---

*Documento elaborado como entregável formal da Etapa 4 — Tech Challenge FIAP Fase 1.*
