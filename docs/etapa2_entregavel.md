# Etapa 2 — Entregável: Modelagem com Redes Neurais


## 1. Objetivo

Construir, treinar e avaliar uma MLP em PyTorch, comparando seu desempenho com os baselines da Etapa 1 (DummyClassifier, LogisticRegression) e com modelos ensemble (Random Forest, Gradient Boosting), usando no mínimo 4 métricas técnicas e análise de custo de negócio (trade-off FP vs FN).

---

## 2. Modelos Treinados

| # | Modelo | Script | Tipo |
|---|---|---|---|
| 1 | DummyClassifier | `02_baselines.py` | Baseline ingênuo |
| 2 | LogisticRegression | `02_baselines.py` | Baseline linear (MVP) |
| 3 | LogisticRegression + Ridge L2 | `02_baselines.py` | Baseline regularizado |
| 4 | Random Forest (200 árvores) | `03_ensemble.py` | Bagging — reduz variância |
| 5 | Gradient Boosting (300 árvores) | `03_ensemble.py` | Boosting — reduz viés |
| 6 | MLP PyTorch (early stopping) | `04_mlp.py` | Rede neural densa |

---

## 3. Arquitetura da MLP

```
Input (23 features)
    → Linear(23 → 64) → ReLU → Dropout(0.3)
    → Linear(64 → 32) → ReLU → Dropout(0.3)
    → Linear(32 → 16) → ReLU → Dropout(0.3)
    → Linear(16 →  1) → Sigmoid
```

**Função de custo:** BCELoss (Binary Cross-Entropy)  
**Otimizador:** Adam (lr=0.001)  
**Épocas máximas:** 150 | **Early stopping patience:** 15 épocas  
**Batch size:** 64 | **Dropout:** 0.3 (regularização)  
**Épocas treinadas:** 26 (early stopping acionado na época 26)  
**Melhor val_loss:** 0.4156  

**Split utilizado:**
- Treino: 4.507 registros (80% do X_train)
- Validação: 1.127 registros (20% do X_train — usado no early stopping)
- Teste: 1.409 registros (intocado, usado apenas na avaliação final)

---

## 4. Tabela Comparativa Final

> Ordenado por AUC-ROC (métrica principal do projeto).

| Modelo | AUC-ROC | Recall | F1 | Precision | Overfitting Gap | Resultado Líquido |
|---|---|---|---|---|---|---|
| **MLP PyTorch** | **0.844** | 0.567 | 0.601 | 0.639 | **0.001** | R$ 172.080 |
| Random Forest | 0.842 | 0.722 | 0.624 | 0.549 | 0.083 ⚠️ | R$ 215.700 |
| LogReg baseline | 0.841 | **0.778** | 0.612 | 0.504 | 0.015 | R$ 230.140 |
| **LogReg Ridge L2** | 0.841 | **0.778** | **0.615** | 0.508 | 0.015 | **R$ 230.340** |
| Gradient Boosting | 0.840 | 0.529 | 0.583 | **0.649** | 0.058 ⚠️ | R$ 160.970 |
| DummyClassifier | 0.500 | 0.000 | 0.000 | 0.000 | — | R$ 0 |

**Metas do projeto:** AUC-ROC ≥ 0.80 ✅ | Recall ≥ 0.70

---

## 5. Análise de Trade-off FP vs FN

**Premissas de negócio adotadas:**

| Premissa | Valor |
|---|---|
| Receita mensal média por cliente | R$ 70,00 |
| LTV estimado (meses de retenção) | 12 meses |
| Receita por churn evitado (VP) | **R$ 840,00** |
| Custo de campanha por cliente abordado (FP) | **R$ 50,00** |

**Definição dos erros:**

- **Falso Negativo (FN):** cliente vai cancelar mas o modelo classificou como "seguro" → empresa não age → **cliente perdido → receita perdida de R$ 840**
- **Falso Positivo (FP):** cliente não vai cancelar mas foi classificado como "em risco" → empresa faz oferta desnecessária → **custo de R$ 50 por abordagem**

> O custo de um FN (R$ 840) é **16,8x maior** que o custo de um FP (R$ 50). Por isso o Recall é priorizado como métrica técnica — identificar o máximo de churners reais é mais valioso do que evitar ofertas desnecessárias.

**Resultado por modelo no conjunto de teste (1.409 clientes):**

| Modelo | VP (detectados) | FN (perdidos) | FP (desnec.) | Receita preservada | Custo FP | **Resultado líquido** |
|---|---|---|---|---|---|---|
| LogReg Ridge L2 | 291 | 83 | 282 | R$ 244.440 | R$ 14.100 | **R$ 230.340** |
| LogReg baseline | 291 | 83 | 286 | R$ 244.440 | R$ 14.300 | **R$ 230.140** |
| Random Forest | 270 | 104 | 222 | R$ 226.800 | R$ 11.100 | **R$ 215.700** |
| MLP PyTorch | 212 | 162 | 120 | R$ 178.080 | R$ 6.000 | **R$ 172.080** |
| Gradient Boosting | 198 | 176 | 107 | R$ 166.320 | R$ 5.350 | **R$ 160.970** |
| DummyClassifier | 0 | 374 | 0 | R$ 0 | R$ 0 | **R$ 0** |

---

## 6. Análise por Modelo

### DummyClassifier
Sempre prediz a classe majoritária (Não churn). AUC-ROC = 0.50 — equivale a chute aleatório. Serve apenas como referência mínima: qualquer modelo real deve superá-lo. Resultado líquido R$ 0 — 374 churners perdidos sem nenhuma ação.

### LogisticRegression (baseline MVP)
Modelo mais simples com sinal preditivo real. AUC-ROC 0.841, Recall 0.778 — já atinge ambas as metas do projeto. Interpretável, sem overfitting (gap=0.015). Bom ponto de partida e referência forte para os modelos mais complexos.

### LogisticRegression + Ridge L2
Melhor resultado líquido de negócio (R$ 230.340). A regularização L2 reduziu levemente o overfitting e os falsos positivos em relação ao baseline, mantendo o mesmo Recall. **Modelo recomendado para produção neste momento.**

### Random Forest (Bagging, 200 árvores)
Maior AUC-ROC em treino (0.937) vs teste (0.842) — gap de 0.083 indica overfitting moderado, esperado em Random Forest com árvores profundas. OOB Score de 0.777 confirma generalização razoável. Feature importance confirma os insights da EDA: `tenure`, `TotalCharges` e `MonthlyCharges` são os 3 fatores mais relevantes para churn.

### Gradient Boosting (Boosting sequencial, 300 árvores)
Maior precisão (0.649) — menos falsos positivos — mas Recall de apenas 0.529, abaixo da meta de 0.70. Priorizou não perturbar clientes não-churners, mas deixou 176 churners sem ação. Configuração sem `class_weight` penalizou o modelo no contexto de dataset desbalanceado. Resultado líquido mais baixo entre os modelos com sinal real.

### MLP PyTorch (early stopping)
**Melhor AUC-ROC geral: 0.844.** Menor overfitting de todos os modelos (gap=0.001 — praticamente zero). Early stopping acionado na época 26 de 150, com melhor val_loss de 0.4156. A BCELoss sem peso de classe tratou as classes igualmente, o que limitou o Recall (0.567) — abaixo da meta. A MLP aprendeu bem a separar as probabilidades (AUC alto), mas o threshold padrão de 0.5 não favoreceu a classe minoritária.

**Oportunidade de melhoria:** ajustar o threshold de classificação (de 0.5 para ~0.35) ou usar BCELoss com peso de classe positiva aumentado pode elevar o Recall da MLP sem retreinamento.

---

## 7. Feature Importance (Random Forest e Gradient Boosting)

Ambos os modelos ensemble concordam nas principais features preditoras de churn:

| Ranking | Feature | RF Importância | GB Importância | Insight |
|---|---|---|---|---|
| 1 | `tenure` | 0.185 | 0.254 | Clientes novos cancelam muito mais |
| 2 | `TotalCharges` | 0.139 | 0.160 | Proxy do LTV — baixo total = risco alto |
| 3 | `MonthlyCharges` | 0.108 | 0.133 | Cobranças altas → insatisfação |
| 4 | `InternetService_Fiber optic` | 0.073 | 0.154 | Fibra tem 42% de churn no dataset |
| 5 | `PaymentMethod_Electronic check` | 0.058 | 0.081 | Método manual → maior churn |
| 6 | `Contract_Two year` | 0.096 | 0.036 | Contrato longo → proteção contra churn |

Esses resultados confirmam os insights da EDA e validam que o modelo aprendeu padrões reais de negócio.

---

## 8. Conclusão e Modelo Recomendado

**Modelo recomendado para produção: LogisticRegression + Ridge L2**

| Critério | Avaliação |
|---|---|
| AUC-ROC | 0.841 ✅ (meta ≥ 0.80) |
| Recall | 0.778 ✅ (meta ≥ 0.70) |
| Overfitting | 0.015 ✅ (sem overfitting) |
| Resultado líquido | R$ 230.340 🥇 (melhor de todos) |
| Interpretabilidade | Alta — coeficientes explicáveis ao negócio |
| Complexidade | Baixa — rápido para treinar e servir via API |

**Por que não a MLP?** A MLP tem o melhor AUC-ROC (0.844) e zero overfitting, mas Recall de 0.567 — abaixo da meta de 0.70 — e resultado líquido R$ 58.260 inferior ao da LogReg Ridge. Uma próxima iteração com ajuste de threshold ou peso de classe pode mudar esse quadro.

**Por que não o Random Forest?** Bom Recall (0.722) mas overfitting de 0.083 e resultado líquido R$ 14.640 inferior ao da LogReg Ridge.

---

## 9. Artefatos no MLflow

Todos os 6 modelos estão registrados no experimento `telco_churn_baselines` com:

- Parâmetros do modelo
- Dataset version: `v1.0-d4ce61f5`
- Métricas: `train_*` e `test_*` para accuracy, AUC-ROC, PR-AUC, F1, Precision, Recall
- Métricas de negócio: VP, FN, FP, receita preservada, resultado líquido
- Artefatos: matriz de confusão, feature importance (ensembles), curva de loss (MLP), checkpoint da MLP (`mlp_best.pt`)

Para visualizar:
```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

## 10. Próximos Passos — Etapa 3

- Refatorar o melhor modelo (`LogisticRegression + Ridge L2`) em `src/` como pipeline serializado
- Construir API de inferência com FastAPI (`/predict`, `/health`)
- Implementar testes automatizados com pytest (smoke test, schema com pandera, teste da API)
- Logging estruturado em toda a aplicação
- Linting com ruff sem erros
- README completo com instruções de setup e execução

---

*Documento gerado como entregável formal da Etapa 2 — Tech Challenge FIAP Fase 1.*