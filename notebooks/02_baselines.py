"""
notebooks/02_baselines.py
Baselines com MLflow — Telco Customer Churn

Segue a estrutura da Ciclo de Vida Aula 02 e Fundamentos Aulas 01/02:
    Modelo 1: DummyClassifier  (baseline ingênuo — referência mínima)
    Modelo 2: LogisticRegression (baseline interpretável — MVP)
    Modelo 3: LogisticRegression + Ridge L2 (regularização)

Todos os experimentos são registrados no MLflow com:
    - Parâmetros do modelo
    - Métricas: accuracy, AUC-ROC, PR-AUC, F1, Precision, Recall
    - Dataset version (nome, shape, hash)
    - Artefato: matriz de confusão

Inclui análise de custo de negócio (métrica de negócio):
    - Falso Negativo: cliente churn não detectado → receita perdida
    - Falso Positivo: cliente não-churn abordado → custo de campanha

Uso:
    python notebooks/02_baselines.py
"""

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import preparar_dados
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
SEED = 42
DATA_PROCESSED = Path("data/processed/telco_clean.csv")
MLFLOW_EXPERIMENT = "telco_churn_baselines"
OUTPUT_DIR = Path("outputs/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Premissas de negócio para análise de custo
# Baseadas no dataset: MonthlyCharges mediana ≈ R$ 70/mês
# LTV estimado = 12 meses de receita média
RECEITA_MENSAL_MEDIA = 70.0       # R$ — mediana do dataset
LTV_MESES = 12                     # meses de receita preservada por retenção
CUSTO_CAMPANHA_RETENCAO = 50.0    # R$ — custo de abordar um cliente (desconto + contato)

RECEITA_POR_CHURN_EVITADO = RECEITA_MENSAL_MEDIA * LTV_MESES   # R$ 840 por FN evitado
CUSTO_POR_FALSO_POSITIVO = CUSTO_CAMPANHA_RETENCAO              # R$ 50 por FP


# ---------------------------------------------------------------------------
# Dataset version
# ---------------------------------------------------------------------------

def obter_info_dataset(caminho: Path) -> dict:
    """
    Gera metadados do dataset para rastreabilidade no MLflow:
    nome, shape, hash MD5 dos primeiros 10MB.
    """
    df = pd.read_csv(caminho)
    conteudo = caminho.read_bytes()[:10_000_000]
    hash_md5 = hashlib.md5(conteudo).hexdigest()[:8]
    return {
        "dataset_name":    caminho.name,
        "dataset_path":    str(caminho),
        "dataset_rows":    df.shape[0],
        "dataset_cols":    df.shape[1],
        "dataset_hash":    hash_md5,
        "dataset_version": f"v1.0-{hash_md5}",
    }


def logar_dataset_version(info: dict) -> None:
    """Loga os metadados do dataset como params no MLflow."""
    for k, v in info.items():
        mlflow.log_param(k, v)
    logger.info("  Dataset version logado: %s", info["dataset_version"])


# ---------------------------------------------------------------------------
# Funções auxiliares de métricas
# ---------------------------------------------------------------------------

def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Calcula e retorna todas as métricas técnicas do projeto."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
    }


def logar_metricas(metricas: dict, prefixo: str = "") -> None:
    """Loga as métricas no logger e no MLflow."""
    for nome, valor in metricas.items():
        chave = f"{prefixo}{nome}" if prefixo else nome
        mlflow.log_metric(chave, valor)
        logger.info("  %-35s %.4f", chave, valor)


def log_overfitting(metricas_treino: dict, metricas_teste: dict) -> None:
    """Calcula e loga o gap treino/teste para detectar overfitting."""
    gap = metricas_treino["accuracy"] - metricas_teste["accuracy"]
    mlflow.log_metric("overfitting_gap_accuracy", gap)
    logger.info("  %-35s %.4f", "overfitting_gap_accuracy", gap)
    if gap > 0.05:
        logger.warning("  Possível overfitting (gap=%.4f > 0.05).", gap)
    else:
        logger.info("  Sem overfitting significativo (gap=%.4f).", gap)


def salvar_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str,
) -> Path:
    """Gera e salva a matriz de confusão como PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matriz de Confusão — {nome_modelo}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    slug = nome_modelo.lower().replace(" ", "_")
    caminho = OUTPUT_DIR / f"confusion_matrix_{slug}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("  Matriz de confusão salva: %s", caminho)
    return caminho


# ---------------------------------------------------------------------------
# Análise de custo de negócio (métrica de negócio)
# ---------------------------------------------------------------------------

def calcular_custo_negocio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str,
) -> dict:
    """
    Calcula a métrica de negócio: custo total e receita preservada.

    Lógica:
        - Falso Negativo (FN): cliente vai cancelar mas modelo errou → receita perdida
        - Falso Positivo (FP): cliente não vai cancelar mas foi abordado → custo de campanha
        - Verdadeiro Positivo (VP): churn detectado → receita preservada pela retenção

    Args:
        y_true: labels reais.
        y_pred: labels preditos.
        nome_modelo: nome para exibição no log.

    Returns:
        Dicionário com métricas de negócio.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    receita_preservada   = tp * RECEITA_POR_CHURN_EVITADO
    custo_falsos_pos     = fp * CUSTO_POR_FALSO_POSITIVO
    receita_perdida      = fn * RECEITA_POR_CHURN_EVITADO
    resultado_liquido    = receita_preservada - custo_falsos_pos

    metricas_negocio = {
        "negocio_vp_churners_detectados":   int(tp),
        "negocio_fn_churners_perdidos":     int(fn),
        "negocio_fp_abordagens_desnec":     int(fp),
        "negocio_receita_preservada_BRL":    receita_preservada,
        "negocio_custo_falsos_pos_BRL":      custo_falsos_pos,
        "negocio_receita_perdida_BRL":       receita_perdida,
        "negocio_resultado_liquido_BRL":     resultado_liquido,
    }

    logger.info("\n  --- Análise de Custo de Negócio — %s ---", nome_modelo)
    logger.info("  Premissas: receita/churn evitado=R$%.0f | custo FP=R$%.0f",
                RECEITA_POR_CHURN_EVITADO, CUSTO_POR_FALSO_POSITIVO)
    logger.info("  Churners detectados (VP):       %d", tp)
    logger.info("  Churners perdidos   (FN):       %d  → receita perdida: R$ %.0f", fn, receita_perdida)
    logger.info("  Abordagens desnec.  (FP):       %d  → custo campanha:  R$ %.0f", fp, custo_falsos_pos)
    logger.info("  Receita preservada  (VP×LTV):   R$ %.0f", receita_preservada)
    logger.info("  Resultado líquido   (pres-cust): R$ %.0f", resultado_liquido)

    for k, v in metricas_negocio.items():
        mlflow.log_metric(k, v)

    return metricas_negocio


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def treinar_dummy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    info_dataset: dict,
) -> None:
    """
    Modelo 1: DummyClassifier (strategy='most_frequent').
    Baseline ingênuo — sempre prediz a classe majoritária.
    Referência mínima: qualquer modelo real deve superar este.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 1 — DummyClassifier (most_frequent)")
    logger.info("=" * 60)

    with mlflow.start_run(run_name="dummy_most_frequent"):
        logar_dataset_version(info_dataset)

        modelo = DummyClassifier(strategy="most_frequent", random_state=SEED)
        modelo.fit(X_train, y_train)

        y_pred_train = modelo.predict(X_train)
        y_pred_test  = modelo.predict(X_test)
        proporcao_churn = y_train.mean()
        y_proba_test = np.full(len(y_test), proporcao_churn)

        mlflow.log_param("strategy", "most_frequent")
        mlflow.log_param("seed", SEED)

        logger.info("  Métricas de treino:")
        metricas_treino = {
            "accuracy":  accuracy_score(y_train, y_pred_train),
            "auc_roc":   0.5,
            "pr_auc":    proporcao_churn,
            "f1":        f1_score(y_train, y_pred_train, zero_division=0),
            "precision": precision_score(y_train, y_pred_train, zero_division=0),
            "recall":    recall_score(y_train, y_pred_train, zero_division=0),
        }
        logar_metricas(metricas_treino, prefixo="train_")

        logger.info("  Métricas de teste:")
        metricas_teste = {
            "accuracy":  accuracy_score(y_test, y_pred_test),
            "auc_roc":   0.5,
            "pr_auc":    proporcao_churn,
            "f1":        f1_score(y_test, y_pred_test, zero_division=0),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall":    recall_score(y_test, y_pred_test, zero_division=0),
        }
        logar_metricas(metricas_teste, prefixo="test_")

        calcular_custo_negocio(y_test, y_pred_test, "DummyClassifier")

        caminho_cm = salvar_matriz_confusao(y_test, y_pred_test, "DummyClassifier")
        mlflow.log_artifact(str(caminho_cm))
        mlflow.sklearn.log_model(modelo, "model")

    logger.info("  Run MLflow registrado: DummyClassifier")


def treinar_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    info_dataset: dict,
) -> None:
    """
    Modelo 2: LogisticRegression (C=1.0, class_weight='balanced').
    Baseline interpretável — MVP conforme Ciclo de Vida Aula 02.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 2 — LogisticRegression (baseline MVP)")
    logger.info("=" * 60)

    params = {
        "C":            1.0,
        "max_iter":     1000,
        "solver":       "lbfgs",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        logar_dataset_version(info_dataset)

        modelo = LogisticRegression(**params)
        modelo.fit(X_train, y_train)

        y_pred_train  = modelo.predict(X_train)
        y_pred_test   = modelo.predict(X_test)
        y_proba_train = modelo.predict_proba(X_train)[:, 1]
        y_proba_test  = modelo.predict_proba(X_test)[:, 1]

        for k, v in params.items():
            mlflow.log_param(k, v)

        logger.info("  Métricas de treino:")
        metricas_treino = calcular_metricas(y_train, y_pred_train, y_proba_train)
        logar_metricas(metricas_treino, prefixo="train_")

        logger.info("  Métricas de teste:")
        metricas_teste = calcular_metricas(y_test, y_pred_test, y_proba_test)
        logar_metricas(metricas_teste, prefixo="test_")

        log_overfitting(metricas_treino, metricas_teste)
        calcular_custo_negocio(y_test, y_pred_test, "LogisticRegression")

        caminho_cm = salvar_matriz_confusao(y_test, y_pred_test, "LogisticRegression")
        mlflow.log_artifact(str(caminho_cm))
        mlflow.sklearn.log_model(modelo, "model")

    logger.info("  Run MLflow registrado: LogisticRegression baseline")


def treinar_logistic_ridge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    info_dataset: dict,
) -> None:
    """
    Modelo 3: LogisticRegression com regularização L2 (Ridge).
    C=0.1 → penalização mais forte, reduz variância dos coeficientes.
    Conforme Fundamentos Aula 02: Ridge estabiliza sob multicolinearidade.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 3 — LogisticRegression + Ridge (L2, C=0.1)")
    logger.info("=" * 60)

    # penalty removido: lbfgs usa L2 por padrão — elimina FutureWarning
    params = {
        "C":            0.1,
        "max_iter":     1000,
        "solver":       "lbfgs",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    with mlflow.start_run(run_name="logistic_regression_ridge_l2"):
        logar_dataset_version(info_dataset)

        mlflow.log_param("regularization", "l2_ridge")
        modelo = LogisticRegression(**params)
        modelo.fit(X_train, y_train)

        y_pred_train  = modelo.predict(X_train)
        y_pred_test   = modelo.predict(X_test)
        y_proba_train = modelo.predict_proba(X_train)[:, 1]
        y_proba_test  = modelo.predict_proba(X_test)[:, 1]

        for k, v in params.items():
            mlflow.log_param(k, v)

        logger.info("  Métricas de treino:")
        metricas_treino = calcular_metricas(y_train, y_pred_train, y_proba_train)
        logar_metricas(metricas_treino, prefixo="train_")

        logger.info("  Métricas de teste:")
        metricas_teste = calcular_metricas(y_test, y_pred_test, y_proba_test)
        logar_metricas(metricas_teste, prefixo="test_")

        log_overfitting(metricas_treino, metricas_teste)
        calcular_custo_negocio(y_test, y_pred_test, "LogisticRegression Ridge L2")

        caminho_cm = salvar_matriz_confusao(y_test, y_pred_test, "LogisticRegression Ridge")
        mlflow.log_artifact(str(caminho_cm))
        mlflow.sklearn.log_model(modelo, "model")

    logger.info("  Run MLflow registrado: LogisticRegression Ridge L2")


# ---------------------------------------------------------------------------
# Tabela comparativa
# ---------------------------------------------------------------------------

def imprimir_tabela_comparativa() -> None:
    """Busca os runs do experimento e imprime tabela comparativa."""
    logger.info("\n%s", "=" * 60)
    logger.info("TABELA COMPARATIVA — Baselines")
    logger.info("=" * 60)

    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        order_by=["metrics.test_auc_roc DESC"],
    )

    if runs.empty:
        logger.warning("Nenhum run encontrado no experimento.")
        return

    colunas_tecnicas = [
        "tags.mlflow.runName",
        "metrics.test_accuracy",
        "metrics.test_auc_roc",
        "metrics.test_pr_auc",
        "metrics.test_f1",
        "metrics.test_recall",
    ]
    colunas_negocio = [
        "metrics.negocio_vp_churners_detectados",
        "metrics.negocio_fn_churners_perdidos",
        "metrics.negocio_resultado_liquido_BRL",
    ]

    colunas = colunas_tecnicas + colunas_negocio
    cols_existentes = [c for c in colunas if c in runs.columns]
    tabela = runs[cols_existentes].copy()
    tabela.columns = [
        c.replace("metrics.", "").replace("tags.mlflow.", "")
        for c in cols_existentes
    ]

    logger.info("\nMétricas técnicas + negócio:\n%s", tabela.to_string(index=False))
    logger.info("\nMetas técnicas:  AUC-ROC >= 0.80 | Recall >= 0.70")
    logger.info("Meta de negócio: Resultado líquido positivo e crescente a cada iteração de modelo.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Baselines — Telco Customer Churn | seed=%d", SEED)

    # Configurar MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT)

    # Metadados do dataset para rastreabilidade
    info_dataset = obter_info_dataset(DATA_PROCESSED)
    logger.info(
        "Dataset: %s | %d linhas | %d colunas | version=%s",
        info_dataset["dataset_name"],
        info_dataset["dataset_rows"],
        info_dataset["dataset_cols"],
        info_dataset["dataset_version"],
    )

    # Preparar dados
    X_train, X_test, y_train, y_test, feature_names = preparar_dados(DATA_PROCESSED)
    logger.info("Primeiras features: %s", feature_names[:5])

    # Treinar modelos
    treinar_dummy(X_train, X_test, y_train, y_test, info_dataset)
    treinar_logistic_regression(X_train, X_test, y_train, y_test, info_dataset)
    treinar_logistic_ridge(X_train, X_test, y_train, y_test, info_dataset)

    # Tabela comparativa
    imprimir_tabela_comparativa()

    logger.info("\nEtapa 1 concluída. Execute 'mlflow ui' para visualizar os experimentos.")


if __name__ == "__main__":
    main()