# ML Canvas — Previsão de Churn em Telecomunicações

**Projeto:** Tech Challenge FIAP — Fase 1  
**Dataset:** IBM Telco Customer Churn  
**Tipo de problema:** Classificação binária supervisionada  
**Data:** Abril/2026  

---

## 1. Objetivo de Negócio

> *"Entender, do ponto de vista de negócio, o que o cliente realmente quer alcançar."*  
> — CRISP-DM, Business Understanding

A operadora de telecomunicações está enfrentando uma alta taxa de cancelamento de clientes (churn), com impacto direto na receita recorrente e na base ativa de assinantes.

**Objetivo:** Reduzir a taxa de churn em **pelo menos 15% no trimestre seguinte ao deploy** do modelo, por meio da identificação antecipada de clientes com risco de cancelamento, viabilizando ações preventivas de retenção antes que o cancelamento ocorra.

**Critério de sucesso do negócio:** A taxa de churn mensal observada na base de clientes atendidos pelo modelo deve cair no mínimo 15% em relação ao trimestre anterior à implementação, medida após 90 dias de operação.

---

## 2. Avaliação da Situação (Assess Situation)

> *"Inventário de recursos, restrições, riscos e contingências."*  
> — CRISP-DM, Business Understanding

### 2.1 Recursos Disponíveis

| Recurso | Descrição |
|---|---|
| **Dados** | Dataset IBM Telco Customer Churn — 7.043 registros, 21 variáveis, histórico de clientes de uma operadora de telecom |
| **Equipe** | Cientistas de dados (equipe do Tech Challenge) |
| **Infraestrutura** | Python local + repositório GitHub |
| **Stack técnica** | PyTorch · Scikit-Learn · MLflow · FastAPI · pytest · ruff |
| **Prazo** | Conforme cronograma do Tech Challenge FIAP Fase 1 |
| **Orçamento** | Sem custo adicional — uso de ferramentas open source |

### 2.2 Restrições

- **Dados:** Apenas o dataset histórico disponível; sem acesso a dados externos (coberturas de sinal, ofertas da concorrência, reclamações em tempo real).
- **Privacidade (LGPD):** Os dados de clientes devem ser tratados como anonimizados. Nenhuma informação de identificação pessoal (PII) pode ser usada como feature no modelo. A coluna `customerID` deve ser excluída do treinamento.
- **Interpretabilidade:** O modelo precisa ser suficientemente explicável para que o time de negócio compreenda os fatores de risco e confie nas previsões ao tomar decisões de retenção.
- **Tecnologia:** Uso obrigatório de PyTorch (MLP), Scikit-Learn (pipeline e baselines) e MLflow (tracking de experimentos), conforme requisitos do Tech Challenge.
- **Reprodutibilidade:** Seeds fixados em todo o pipeline; instalação do zero via `pyproject.toml`.

### 2.3 Pressupostos

- O padrão histórico de churn registrado no dataset é representativo do comportamento atual dos clientes.
- A variável `Churn` (Yes/No) está corretamente rotulada e reflete cancelamentos reais, não apenas inatividade.
- Os dados disponíveis têm qualidade suficiente para suportar um modelo preditivo útil após o tratamento adequado na EDA.
- O time de retenção terá condições operacionais de agir sobre a lista de clientes em risco gerada pelo modelo.

### 2.4 Riscos e Contingências

| Risco | Probabilidade | Impacto | Contingência |
|---|---|---|---|
| Desbalanceamento de classes (~26% churn) | Alta | Alto | Usar validação estratificada, métricas adequadas (AUC-ROC, PR-AUC) e avaliar técnicas de balanceamento (SMOTE, class_weight) |
| `TotalCharges` com valores não numéricos (espaços em branco) | Confirmada na EDA | Médio | Converter para float e imputar com mediana os registros com valor vazio |
| Multicolinearidade entre `tenure`, `MonthlyCharges` e `TotalCharges` | Alta | Médio | Analisar correlação na EDA e aplicar pipeline de pré-processamento padronizado |
| Overfitting da MLP em dataset pequeno (7k registros) | Média | Alto | Implementar early stopping, dropout e validação cruzada estratificada |
| Dados não suficientes para generalizar para perfis fora do histórico | Média | Médio | Documentar limitações no Model Card; não extrapolar para perfis muito distintos dos dados de treino |

---

## 3. Objetivos de Data Science

> *"Traduzir o objetivo de negócio em objetivos técnicos mensuráveis."*  
> — CRISP-DM, Business Understanding

**Tarefa de ML:** Classificação binária supervisionada — dado o perfil atual de um cliente (contrato, serviços, dados demográficos, histórico de pagamentos), prever se ele irá cancelar o serviço (`Churn = Yes`) ou não (`Churn = No`).

**Variável alvo (target):** `Churn` — binária (Yes/No), transformada em 1/0 no pré-processamento.

**Definição operacional de churn:** Cliente que cancelou formalmente o serviço, conforme registrado na coluna `Churn` do dataset. Não inclui inatividade parcial.

### 3.1 Métricas Técnicas

| Métrica | Meta mínima | Justificativa |
|---|---|---|
| **AUC-ROC** | ≥ 0.80 | Métrica principal; robusta ao desbalanceamento de classes |
| **Recall (classe churn)** | ≥ 0.70 | Priorizado: custo de falso negativo (cliente sai sem ação) supera o custo de falso positivo (oferta desnecessária enviada a quem não ia cancelar) |
| **PR-AUC** | ≥ 0.65 | Complementa o AUC-ROC em cenários desbalanceados |
| **F1-Score** | ≥ 0.68 | Equilíbrio entre precisão e recall |
| **Acurácia balanceada** | ≥ 0.75 | Referência geral, dado o desbalanceamento |

### 3.2 Cadeia Lógica Métrica Técnica → KPI de Negócio

```
AUC-ROC ≥ 0.80
    → identificar ≥ 70% dos clientes que irão cancelar (Recall ≥ 0.70)
        → acionar campanhas de retenção focadas nesses clientes
            → redução observada de churn ≥ 15% no trimestre seguinte
```

### 3.3 Entregáveis Técnicos

- Notebook de EDA com análise completa de qualidade, distribuição e data readiness
- Baselines registrados no MLflow: `DummyClassifier` e `LogisticRegression`
- MLP em PyTorch com early stopping, treinada e avaliada com ≥ 4 métricas
- Tabela comparativa de modelos (baselines + MLP + modelos de ensemble)
- API de inferência em FastAPI com endpoint `/predict` e `/health`
- Model Card com limitações, vieses e plano de monitoramento
- Repositório GitHub com estrutura organizada (`src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`)

---

## 4. Stakeholders

> *"Quem envolver e por quê — desde o Dia 1."*  
> — Ciclo de Vida, Aula 01

| Papel | Representante (contexto do projeto) | Responsabilidade no projeto |
|---|---|---|
| **Sponsor executivo** | Diretoria de Negócios / Operações | Valida o objetivo de negócio, define a meta de redução de churn (15%), aprova recursos e destravar acesso a dados |
| **Especialistas de domínio** | Equipe de CRM e Retenção | Interpretam variáveis, validam hipóteses sobre causas de churn, confirmam se os padrões encontrados no modelo fazem sentido no mundo real |
| **Donos dos dados (Data owners)** | TI / Engenharia de Dados | Administram as bases de clientes, garantem qualidade dos dados, alertam sobre limitações e questões de privacidade (LGPD) |
| **Usuários finais / Operacionais** | Time de Sucesso do Cliente | Receberão a lista priorizada de clientes em risco e executarão as ações de retenção; definem o formato esperado de entrega (ex.: relatório semanal, dashboard) |
| **Equipe técnica** | Cientistas de Dados / MLOps (equipe FIAP) | Desenvolve, treina, avalia e monitora o modelo; responsável pela API, testes e documentação |

**Como envolver:**
- **Kick-off inicial:** reunião com todos os stakeholders para nivelar entendimento, registrar acordos sobre objetivo, métricas de sucesso e formato de entrega.
- **Checkpoints regulares:** após EDA, após primeiros baselines e após treinamento da MLP — apresentar achados e coletar feedback dos especialistas de domínio.
- **Documentação compartilhada:** atas de decisão e o próprio ML Canvas validado com os patrocinadores do projeto.
- **Glossário comum:** alinhar termos técnicos (AUC-ROC, recall, falso positivo) em linguagem compreensível ao time de negócio antes das apresentações.

---

## 5. Requisitos, Restrições e Expectativas

> *"O contrato social entre a equipe de data science e os stakeholders."*  
> — Ciclo de Vida, Aula 01

### 5.1 Requisitos

- O modelo deve prever o risco de churn de cada cliente e retornar uma probabilidade (0 a 1) e uma classificação binária (em risco / seguro).
- O pipeline completo deve ser reprodutível: qualquer pessoa deve conseguir instalar e executar do zero via `pyproject.toml`.
- A API de inferência deve retornar a predição em tempo real (endpoint `/predict`) e estar disponível para consulta pelo time operacional.
- O modelo deve ser avaliado com no mínimo 4 métricas (AUC-ROC, Recall, F1, PR-AUC) usando validação cruzada estratificada.
- Testes automatizados obrigatórios: no mínimo 3 (smoke test, schema com pandera, teste da API).
- Logging estruturado em toda a aplicação (sem uso de `print()`).
- Linting com `ruff` sem erros.
- Seeds fixados para reprodutibilidade total.

### 5.2 Restrições

- Apenas dados do dataset IBM Telco Customer Churn — sem enriquecimento com fontes externas.
- Nenhuma PII pode ser usada como feature (`customerID` excluído).
- Stack obrigatória definida pelo Tech Challenge: PyTorch, Scikit-Learn, MLflow, FastAPI.
- Prazo fixo conforme calendário acadêmico da FIAP.
- Equipe reduzida — divisão de responsabilidades entre membros do grupo.

### 5.3 Expectativas dos Stakeholders

| Stakeholder | Expectativa |
|---|---|
| Diretoria | Redução mensurável de churn após 90 dias de operação do modelo; relatório executivo com impacto financeiro estimado |
| Equipe de CRM | Lista semanal dos clientes com maior probabilidade de churn, ordenada por score de risco, para priorizar contato |
| TI / Engenharia | API estável, documentada e com testes; modelo versionado e rastreável via MLflow |
| Usuários finais | Formato de entrega simples e acionável — não querem caixa-preta; querem entender por que um cliente foi classificado como em risco |

---

## 6. Variáveis e KPIs

> *"Conectar o mundo do negócio (indicadores de sucesso) com o mundo dos dados (atributos mensuráveis)."*  
> — Ciclo de Vida, Aula 01

### 6.1 KPIs de Negócio

| KPI | Descrição | Meta |
|---|---|---|
| **Taxa de churn mensal** | Percentual de clientes que cancelam o serviço a cada mês | Reduzir ≥ 15% em relação ao baseline pré-modelo |
| **Custo de churn evitado** | Receita recorrente preservada pela retenção de clientes que seriam perdidos | Acompanhar após deploy (receita média por cliente × churners retidos) |
| **Taxa de retenção pós-ação** | Percentual de clientes em risco que, após abordagem, permaneceram na base | Aumentar vs. campanha sem modelo (baseline operacional) |
| **ROI da campanha de retenção** | Retorno sobre o investimento nas ações de retenção direcionadas pelo modelo | Positivo no primeiro trimestre de operação |

### 6.2 Variável Alvo

- **Nome:** `Churn`
- **Tipo:** Binária — `Yes` (cancelou) / `No` (permaneceu)
- **Transformação:** Codificada como `1` (churn) e `0` (não churn) no pré-processamento
- **Distribuição esperada:** ~26% positiva (churn) — dataset desbalanceado, exige atenção nas métricas e estratégia de treinamento

### 6.3 Features Disponíveis no Dataset

**Contínuas / Numéricas:**

| Feature | Descrição | Relevância esperada |
|---|---|---|
| `tenure` | Tempo em meses como cliente | Alta — clientes com menos tempo têm maior propensão a churn |
| `MonthlyCharges` | Valor da fatura mensal | Alta — cobranças elevadas correlacionam com insatisfação |
| `TotalCharges` | Valor total gasto pelo cliente (string com ruído — requer tratamento) | Alta — proxy do LTV do cliente |

**Categóricas de Contrato e Pagamento:**

| Feature | Descrição | Relevância esperada |
|---|---|---|
| `Contract` | Tipo de contrato (Month-to-month, One year, Two year) | Muito alta — contratos mensais têm churn muito mais alto |
| `PaymentMethod` | Método de pagamento (Cartão, Boleto, Débito automático) | Alta — métodos manuais associados a mais churn |
| `PaperlessBilling` | Faturamento sem papel (Yes/No) | Média |
| `InternetService` | Tipo de serviço de internet (DSL, Fiber optic, No) | Alta — fibra ótica tem maior churn no dataset |

**Serviços Contratados (categóricas binárias):**

| Feature | Descrição |
|---|---|
| `PhoneService` | Possui serviço telefônico? |
| `MultipleLines` | Possui múltiplas linhas? |
| `OnlineSecurity` | Possui segurança online? |
| `OnlineBackup` | Possui backup online? |
| `DeviceProtection` | Possui proteção de dispositivo? |
| `TechSupport` | Possui suporte técnico? |
| `StreamingTV` | Possui TV por streaming? |
| `StreamingMovies` | Possui filmes por streaming? |

**Perfil Demográfico:**

| Feature | Descrição | Observação |
|---|---|---|
| `gender` | Gênero do cliente | Monitorar viés — verificar impacto na EDA |
| `SeniorCitizen` | Idoso? (0/1) | Grupo específico com comportamento distinto |
| `Partner` | Possui parceiro(a)? | Fator de estabilidade contratual |
| `Dependents` | Possui dependentes? | Fator de estabilidade contratual |

**Excluída do modelo:**

| Feature | Motivo |
|---|---|
| `customerID` | Identificador único — não tem poder preditivo e é PII |

### 6.4 Categorização das Features (modelo mental da aula)

| Categoria | Features |
|---|---|
| **Demográficas** (quem é o cliente) | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Comportamentais** (como usa o serviço) | `PhoneService`, `MultipleLines`, `InternetService`, serviços adicionais |
| **Históricas** (passado de relacionamento) | `tenure`, `TotalCharges`, `Contract` |
| **Contextuais** (condições do contrato) | `MonthlyCharges`, `PaymentMethod`, `PaperlessBilling` |

---

## 7. Plano de Projeto

> *"Um mapa para guiar o projeto — cronograma, responsáveis, dependências e ferramentas."*  
> — CRISP-DM, Business Understanding

| Etapa | Foco | Entregável |
|---|---|---|
| **Etapa 1** — Entendimento e Preparação | EDA completa + ML Canvas + Baselines (DummyClassifier, LogisticRegression) registrados no MLflow | Notebook de EDA + baselines no MLflow |
| **Etapa 2** — Modelagem com Redes Neurais | MLP em PyTorch com early stopping; comparação com baselines e ensemble; análise de custo (FP vs FN) | Tabela comparativa de modelos + MLP treinado + artefatos no MLflow |
| **Etapa 3** — Engenharia e API | Refatoração em módulos (`src/`), pipeline reprodutível, API FastAPI (`/predict`, `/health`), testes com pytest, logging, `pyproject.toml`, Makefile | Repositório refatorado + API funcional + testes passando |
| **Etapa 4** — Documentação e Entrega Final | Model Card, README completo, plano de monitoramento, vídeo STAR de 5 minutos | Repositório final + vídeo STAR + (opcional) deploy em nuvem |

**Dependências críticas:**
- A Etapa 2 depende da limpeza e feature engineering definidos na Etapa 1.
- A Etapa 3 depende do modelo finalizado na Etapa 2 para servir via API.
- A Etapa 4 depende de todas as anteriores.

**Ferramentas definidas:**

| Finalidade | Ferramenta |
|---|---|
| Modelagem (rede neural) | PyTorch |
| Pipeline e baselines | Scikit-Learn |
| Tracking de experimentos | MLflow |
| API de inferência | FastAPI + Pydantic |
| Testes | pytest + pandera |
| Qualidade de código | ruff |
| Gerenciamento de dependências | pyproject.toml |
| Versionamento | Git + GitHub |

---

## 8. SLOs — Objetivos de Nível de Serviço

> *"Requisitos de desempenho operacional que o modelo deve cumprir em produção."*

| SLO | Definição |
|---|---|
| **Modo de inferência** | Batch semanal — lista de clientes em risco gerada toda segunda-feira |
| **Latência de entrega** | Resultados disponíveis em até 24h após coleta dos dados de entrada |
| **Disponibilidade da API** | ≥ 99% do tempo em horário comercial (9h–18h, dias úteis) |
| **Monitoramento de drift** | PSI (Population Stability Index) calculado mensalmente por feature; alerta se PSI > 0.2 em qualquer variável relevante |
| **Alerta de degradação** | Alerta automático se AUC-ROC cair mais de 5 pontos percentuais em janela de 30 dias |
| **Retreinamento** | Programado trimestralmente ou quando alerta de drift/degradação for disparado |
| **Reprodutibilidade** | Seeds fixados; pipeline instala e executa do zero via `pyproject.toml` em qualquer máquina |

---

## 9. Glossário

| Termo | Definição |
|---|---|
| **Churn** | Cancelamento formal do serviço pelo cliente |
| **AUC-ROC** | Área sob a curva ROC — mede a capacidade do modelo de separar churners de não-churners em diferentes limiares de classificação |
| **Recall (sensibilidade)** | Proporção de churners reais que o modelo identifica corretamente — minimiza falsos negativos |
| **Falso negativo** | Cliente que vai cancelar mas o modelo classificou como "seguro" — o custo mais alto neste projeto |
| **Falso positivo** | Cliente que não vai cancelar mas o modelo classificou como "em risco" — gera custo de oferta desnecessária, mas é menos grave |
| **PR-AUC** | Área sob a curva Precisão-Recall — métrica robusta para datasets desbalanceados |
| **PSI** | Population Stability Index — mede se a distribuição das features mudou entre treino e produção |
| **MLP** | Multi-Layer Perceptron — rede neural densa, arquitetura central do projeto |
| **MLflow** | Plataforma de tracking de experimentos — registra parâmetros, métricas e artefatos de cada rodada de treinamento |
| **Early stopping** | Técnica de interromper o treinamento quando a perda de validação para de melhorar, prevenindo overfitting |
| **LGPD** | Lei Geral de Proteção de Dados — exige anonimização e cuidado no uso de dados pessoais de clientes |
| **LTV** | Lifetime Value — valor total que um cliente gera para a empresa ao longo do relacionamento |

---

*Documento elaborado com base na metodologia CRISP-DM conforme Ciclo de Vida, Aula 01 — Machine Learning Engineering, FIAP Fase 1.*