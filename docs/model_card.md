# Model Card — Telco Customer Churn Prediction

**Projeto:** Tech Challenge FIAP — Fase 1  
**Versão do modelo:** 1.0.0  
**Data:** Maio/2026  
**Autores:** Equipe Tech Challenge FIAP  

---

## 1. Detalhes do Modelo

| Campo | Valor |
|---|---|
| **Nome** | Telco Churn Classifier v1.0 |
| **Tipo** | LogisticRegression + Ridge L2 (sklearn) |
| **Tarefa** | Classificação binária supervisionada |
| **Parâmetros** | C=0.1, solver=lbfgs, class_weight=balanced, max_iter=1000, random_state=42 |
| **Pipeline** | ColumnTransformer (StandardScaler + OneHotEncoder) → LogisticRegression |
| **Serialização** | `models/churn_pipeline.joblib` (joblib) |
| **Versão sklearn** | ≥ 1.4 |
| **Dataset de treino** | IBM Telco Customer Churn — 7.043 registros |

---

## 2. Uso Pretendido

### 2.1 Uso previsto

O modelo foi desenvolvido para prever a probabilidade de cancelamento (churn) de clientes de uma operadora de telecomunicações, com base no perfil de contrato, serviços contratados e dados demográficos.

**Casos de uso pretendidos:**

- Geração semanal de lista priorizada de clientes em risco de churn para a equipe de retenção
- Pontuação de risco em tempo real via API para sistemas CRM
- Apoio à tomada de decisão em campanhas preventivas de retenção

**Usuários pretendidos:**

- Equipes de CRM e Sucesso do Cliente da operadora
- Sistemas de automação de marketing
- Analistas de negócio com acesso à API

### 2.2 Usos fora do escopo

- **Não deve ser usado** para decisões que afetem diretamente direitos de clientes (recusa de serviço, crédito, cancelamento forçado)
- **Não deve ser usado** como único critério para corte de benefícios ou serviços
- **Não deve ser usado** para perfis de clientes muito diferentes dos dados de treino (ex.: segmento corporativo, clientes fora do Brasil)
- **Não deve ser usado** em cenários onde o custo de falso positivo supere o custo de falso negativo (o modelo prioriza Recall)

---

## 3. Dados de Treino

| Aspecto | Detalhe |
|---|---|
| **Dataset** | IBM Telco Customer Churn (Kaggle) |
| **Registros** | 7.043 clientes |
| **Split** | 80% treino / 20% teste (estratificado por Churn) |
| **Período** | Dataset histórico — sem data de coleta explícita |
| **Taxa de churn** | ~26,5% (desbalanceamento moderado) |
| **PII removida** | `customerID` excluído antes do treinamento |

### Features utilizadas (19 campos)

| Categoria | Features |
|---|---|
| **Numéricas** (StandardScaler) | `tenure`, `MonthlyCharges`, `TotalCharges` |
| **Nominais** (OneHotEncoder, drop=first) | `InternetService`, `Contract`, `PaymentMethod` |
| **Binárias** (0/1) | `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` |
| **Ordinais de serviço** (0/1/2) | `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 4. Métricas de Desempenho

### 4.1 Resultados no conjunto de teste (1.409 clientes)

| Modelo | AUC-ROC | Recall | F1 | Precision | Overfitting Gap | Resultado Líquido |
|---|---|---|---|---|---|---|
| **LogReg Ridge L2** ✅ | **0.841** | **0.778** | **0.615** | 0.508 | 0.015 | **R$ 230.340** |
| LogReg baseline | 0.841 | 0.778 | 0.612 | 0.504 | 0.015 | R$ 230.140 |
| Random Forest | 0.842 | 0.722 | 0.624 | 0.549 | 0.083 ⚠️ | R$ 215.700 |
| MLP PyTorch | 0.844 | 0.567 | 0.601 | 0.639 | 0.001 | R$ 172.080 |
| Gradient Boosting | 0.840 | 0.529 | 0.583 | 0.649 | 0.058 ⚠️ | R$ 160.970 |
| DummyClassifier | 0.500 | 0.000 | 0.000 | 0.000 | — | R$ 0 |

### 4.2 Desempenho no conjunto de teste — modelo escolhido

| Métrica | Meta | Resultado | Status |
|---|---|---|---|
| AUC-ROC | ≥ 0.80 | 0.841 | ✅ |
| Recall (classe churn) | ≥ 0.70 | 0.778 | ✅ |
| PR-AUC | ≥ 0.65 | — | — |
| F1-Score | ≥ 0.68 | 0.615 | ⚠️ |
| Overfitting gap | < 0.05 | 0.015 | ✅ |

### 4.3 Análise de custo de negócio (teste — 1.409 clientes)

| Tipo | Quantidade | Impacto financeiro |
|---|---|---|
| Verdadeiros Positivos (VP) | 291 churners detectados | R$ 244.440 preservados |
| Falsos Negativos (FN) | 83 churners não detectados | Receita perdida |
| Falsos Positivos (FP) | 282 abordagens desnecessárias | R$ 14.100 em custo |
| **Resultado líquido** | — | **R$ 230.340** |

**Premissas:**  
- Receita por churn evitado (LTV 12 meses × R$ 70/mês): **R$ 840**  
- Custo de campanha por cliente abordado: **R$ 50**  
- O custo de um FN é **16,8× maior** que o custo de um FP → Recall priorizado

### 4.4 Threshold de decisão

O threshold padrão de **0.5** é usado na predição binária. O campo `risco` da API classifica em três faixas:

| Faixa de probabilidade | Risco |
|---|---|
| ≥ 0.6 | Alto |
| 0.4 – 0.6 | Médio |
| < 0.4 | Baixo |

---

## 5. Análise de Features Relevantes

Com base nos modelos de ensemble (Random Forest e Gradient Boosting), as features mais preditivas são:

| Ranking | Feature | Insight |
|---|---|---|
| 1 | `tenure` | Clientes novos (< 6 meses) cancelam muito mais — maior fator de risco |
| 2 | `TotalCharges` | Proxy do LTV — baixo total = relacionamento curto = risco alto |
| 3 | `MonthlyCharges` | Cobranças altas correlacionam com insatisfação |
| 4 | `InternetService: Fiber optic` | Fibra ótica tem 42% de churn no dataset |
| 5 | `PaymentMethod: Electronic check` | Método manual associado a maior churn |
| 6 | `Contract: Two year` | Contratos longos protegem contra churn |

---

## 6. Limitações

### 6.1 Limitações dos dados

- **Dataset estático:** Treinado em um corte histórico único; sem dados de reclamações em tempo real, cobertura de sinal, ou ofertas da concorrência que possam influenciar o churn atual.
- **Escopo geográfico:** IBM Telco dataset — contexto norte-americano. Padrões de comportamento podem diferir em outras regiões.
- **Período de coleta indefinido:** Sem data explícita de coleta; pode não refletir comportamentos pós-pandemia ou mudanças de mercado recentes.
- **Ausência de dados de comportamento:** Sem histórico de chamadas de suporte, uso de dados, ou interações com o atendimento ao cliente.
- **Clientes novos:** O modelo não foi treinado com dados de clientes com < 1 mês de tenure — predições nesse intervalo devem ser interpretadas com cautela.

### 6.2 Limitações do modelo

- **Threshold fixo:** O threshold padrão de 0.5 não foi otimizado por custo de negócio; ajustá-lo para ~0.35 poderia aumentar o Recall às custas de mais FP.
- **F1-Score abaixo da meta:** F1=0.615 (meta ≥ 0.68) — o modelo prioriza Recall e tem precisão moderada (50.8%); aproximadamente metade das abordagens são FP.
- **Sem explicação por cliente:** O modelo retorna probabilidade e risco, mas não explica quais features contribuíram para aquele cliente específico (sem SHAP integrado).
- **Generalização temporal:** O modelo foi treinado e avaliado no mesmo período histórico — não há validação temporal (treino em T1, teste em T2).

### 6.3 Distribuições representadas no treino

O modelo tem melhor desempenho para perfis similares aos presentes no dataset de treino:

- Clientes com contrato Month-to-month e Fiber optic (alto churn histórico)
- Clientes com tenure entre 1 e 72 meses
- Clientes com MonthlyCharges entre R$18 e R$118

Perfis fora dessas distribuições devem ser tratados com cautela extra.

---

## 7. Análise de Vieses (Bias)

### 7.1 Features demográficas presentes

O dataset inclui `gender` e `SeniorCitizen` como features preditoras. Isso implica que o modelo pode, diretamente, produzir predições diferentes para grupos demográficos.

| Feature demográfica | Impacto potencial |
|---|---|
| `gender` (Male/Female → 0/1) | O modelo usa gênero como input; predições podem diferir sistematicamente por gênero |
| `SeniorCitizen` (0/1) | Clientes idosos têm comportamento distinto no dataset — feature com poder preditivo real, mas grupo protegido pela LGPD |
| `Partner` / `Dependents` | Proxies de estabilidade familiar — podem correlacionar com status socioeconômico |

### 7.2 Avaliação de paridade (recomendada, não realizada)

Para uso em produção completa, recomenda-se calcular as seguintes métricas por subgrupo demográfico:

- **Demographic Parity:** P(predição=churn | gênero=F) ≈ P(predição=churn | gênero=M)?
- **Equalized Odds:** Recall e FPR iguais por gênero e faixa etária?
- **Calibração:** As probabilidades preditas refletem taxas reais de churn por subgrupo?

### 7.3 Decisão adotada

A equipe optou por manter `gender` e `SeniorCitizen` no modelo por terem poder preditivo e estarem dentro do escopo do dataset de treinamento fornecido (IBM Telco). Em produção, deve-se monitorar a distribuição de scores por subgrupo e, se houver disparidade, considerar:

1. Remover as features demográficas e avaliar impacto no AUC-ROC
2. Aplicar técnicas de fairness-aware training (ex.: `fairlearn`)
3. Auditar periodicamente os resultados das campanhas por subgrupo

---

## 8. Cenários de Falha

### 8.1 Falhas previsíveis e mitigações

| Cenário | Causa provável | Impacto | Mitigação |
|---|---|---|---|
| **AUC-ROC cai abaixo de 0.78** | Data drift — perfil de clientes mudou | Modelo produz score não confiável | Alerta automático → análise → retreinamento |
| **Recall cai abaixo de 0.65** | Mudança na taxa de churn da base, novos planos | Churners não detectados → perda de receita | PSI por feature > 0.2 → retreino programado |
| **API retorna 503** | Pipeline não carregado no startup | Indisponibilidade total do serviço | Health check `/health` + rollback para versão anterior |
| **Input com feature ausente** | Bug no sistema de origem dos dados | Erro 422 (validação Pydantic) | Schema strict com defaults documentados |
| **Todos os clientes classificados como "Alto Risco"** | Threshold mal calibrado ou modelo degradado | Equipe de retenção sobrecarregada | Monitorar distribuição de scores periodicamente |
| **Predições inconsistentes entre execuções** | Versão de sklearn diferente | Resultados não-reprodutíveis | Fixar versão no `pyproject.toml`; usar `models/churn_pipeline.joblib` |

### 8.2 Limitações críticas em produção

- O modelo **não detecta churn por motivos externos** (fechamento da empresa, mudança de cidade, óbito) — esses eventos são aleatórios e não preditíveis pelo perfil histórico.
- Em campanhas de retenção muito agressivas, a base de clientes pode mudar de comportamento, invalidando os padrões aprendidos — fenômeno conhecido como **concept drift** induzido por intervenção.
- Clientes que já decidiram cancelar e estão no período de aviso prévio podem ser identificados, mas a janela de ação é curta.

---

## 9. Rastreabilidade e Reprodutibilidade

| Artefato | Localização |
|---|---|
| Pipeline treinado | `models/churn_pipeline.joblib` |
| Checkpoint MLP | `models/mlp_best.pt` |
| Experimentos MLflow | `mlruns/` (local) |
| Código de treino | `src/pipeline.py` |
| Testes automatizados | `tests/` (pytest + pandera) |
| Dependências fixadas | `pyproject.toml` |

**Reprodução do treinamento:**

```bash
make install
make train
```

**Todos os seeds fixados em 42** (`random_state=42` em todos os modelos sklearn; `torch.manual_seed(42)` na MLP).

---

## 10. Recomendações de Uso Responsável

1. **Auditoria regular:** Revisar distribuição de predições por subgrupo demográfico trimestralmente
2. **Supervisão humana:** A lista de clientes em risco deve ser revisada pela equipe de CRM antes de ações de retenção
3. **Transparência:** Informar internamente que decisões de retenção são assistidas por modelo de ML
4. **Retreinamento:** Retreinar com dados mais recentes a cada 3 meses ou quando drift for detectado
5. **Versionamento:** Manter histórico de versões do modelo com métricas de avaliação por versão

---

*Model Card elaborado conforme boas práticas de documentação de modelos (Mitchell et al., 2019) e requisitos do Tech Challenge FIAP Fase 1 — Etapa 4.*
