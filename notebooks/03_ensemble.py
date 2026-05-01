"""
notebooks/03_ensemble.py
Modelos Ensemble — Telco Customer Churn

Baseado em Fundamentos, Aula 04:
    Modelo 1: Random Forest  (Bagging — reduz variância)
    Modelo 2: Gradient Boosting  (Boosting — reduz viés)

Todos os experimentos registrados no MLflow com:
    - Parâmetros, métricas, dataset version
    - Feature importance (top 10)
    - Análise de custo de negócio (FP vs FN)
    - Comparação com baselines da Etapa 1

Uso:
    python notebooks/03_ensemble.py
"""

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
DATA_PROCESSED    = Path("data/processed/telco_clean.csv")
MLFLOW_EXPERIMENT = "telco_churn_baselines"
OUTPUT_DIR        = Path("outputs/ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Premissas de negócio (mesmas do 02_baselines.py)
RECEITA_MENSAL_MEDIA       = 70.0
LTV_MESES                  = 12
CUSTO_CAMPANHA_RETENCAO    = 50.0
RECEITA_POR_CHURN_EVITADO  = RECEITA_MENSAL_MEDIA * LTV_MESES
CUSTO_POR_FALSO_POSITIVO   = CUSTO_CAMPANHA_RETENCAO


# ---------------------------------------------------------------------------
# Utilitários (mesmos do 02_baselines.py — mantém consistência)
# ---------------------------------------------------------------------------

def obter_info_dataset(caminho: Path) -> dict:
    df      = pd.read_csv(caminho)
    conteudo = caminho.read_bytes()[:10_000_000]
    hash_md5 = hashlib.md5(conteudo).hexdigest()[:8]
    return {
        "dataset_name":    caminho.name,
        "dataset_rows":    df.shape[0],
        "dataset_cols":    df.shape[1],
        "dataset_version": f"v1.0-{hash_md5}",
    }


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "auc_roc":   roc_auc_score(y_true, y_proba),
        "pr_auc":    average_precision_score(y_true, y_proba),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
    }


def logar_metricas(metricas: dict, prefixo: str = "") -> None:
    for nome, valor in metricas.items():
        chave = f"{prefixo}{nome}" if prefixo else nome
        mlflow.log_metric(chave, valor)
        logger.info("  %-35s %.4f", chave, valor)


def log_overfitting(metricas_treino: dict, metricas_teste: dict) -> None:
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
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Não churn", "Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Matriz de Confusão — {nome_modelo}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    slug    = nome_modelo.lower().replace(" ", "_")
    caminho = OUTPUT_DIR / f"confusion_matrix_{slug}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("  Matriz de confusão salva: %s", caminho)
    return caminho


def calcular_custo_negocio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome_modelo: str,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    receita_preservada = tp * RECEITA_POR_CHURN_EVITADO
    custo_fp           = fp * CUSTO_POR_FALSO_POSITIVO
    receita_perdida    = fn * RECEITA_POR_CHURN_EVITADO
    resultado_liquido  = receita_preservada - custo_fp

    logger.info("\n  --- Custo de Negócio — %s ---", nome_modelo)
    logger.info("  Churners detectados (VP): %d", tp)
    logger.info("  Churners perdidos   (FN): %d  → receita perdida: R$ %.0f", fn, receita_perdida)
    logger.info("  Abordagens desnec.  (FP): %d  → custo campanha:  R$ %.0f", fp, custo_fp)
    logger.info("  Receita preservada       : R$ %.0f", receita_preservada)
    logger.info("  Resultado líquido        : R$ %.0f", resultado_liquido)

    metricas = {
        "negocio_vp_churners_detectados": int(tp),
        "negocio_fn_churners_perdidos":   int(fn),
        "negocio_fp_abordagens_desnec":   int(fp),
        "negocio_receita_preservada_BRL": receita_preservada,
        "negocio_receita_perdida_BRL":    receita_perdida,
        "negocio_resultado_liquido_BRL":  resultado_liquido,
    }
    for k, v in metricas.items():
        mlflow.log_metric(k, v)
    return metricas


def salvar_feature_importance(
    importancias: np.ndarray,
    feature_names: list[str],
    nome_modelo: str,
    top_n: int = 10,
) -> Path:
    """Plota e salva o gráfico de importância das features (top N)."""
    indices = np.argsort(importancias)[::-1][:top_n]
    nomes   = [feature_names[i] for i in indices]
    valores = importancias[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(nomes[::-1], valores[::-1], color="#378ADD")
    ax.set_xlabel("Importância")
    ax.set_title(f"Top {top_n} Features — {nome_modelo}", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    slug    = nome_modelo.lower().replace(" ", "_")
    caminho = OUTPUT_DIR / f"feature_importance_{slug}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("  Feature importance salva: %s", caminho)

    logger.info("  Top %d features:", top_n)
    for nome, val in zip(nomes, valores):
        logger.info("    %-35s %.4f", nome, val)

    return caminho


# ---------------------------------------------------------------------------
# Modelo 1 — Random Forest
# ---------------------------------------------------------------------------

def treinar_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    info_dataset: dict,
) -> None:
    """
    Random Forest com 200 árvores.
    Bagging: cada árvore treina em bootstrap diferente,
    seleção aleatória de features em cada nó → reduz variância.
    Conforme Fundamentos, Aula 04.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 4 — Random Forest (Bagging, 200 árvores)")
    logger.info("=" * 60)

    params = {
        "n_estimators":    200,
        "max_features":    "sqrt",   # sqrt(d) — padrão recomendado por Breiman
        "min_samples_leaf": 5,       # evita divisões muito específicas
        "class_weight":    "balanced",
        "oob_score":       True,     # estimativa out-of-bag sem validação separada
        "n_jobs":          -1,
        "random_state":    SEED,
    }

    with mlflow.start_run(run_name="random_forest_200"):
        for k, v in info_dataset.items():
            mlflow.log_param(k, v)

        modelo = RandomForestClassifier(**params)
        modelo.fit(X_train, y_train)

        y_pred_train  = modelo.predict(X_train)
        y_pred_test   = modelo.predict(X_test)
        y_proba_train = modelo.predict_proba(X_train)[:, 1]
        y_proba_test  = modelo.predict_proba(X_test)[:, 1]

        for k, v in params.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("oob_score", modelo.oob_score_)
        logger.info("  OOB Score: %.4f", modelo.oob_score_)

        logger.info("  Métricas de treino:")
        metricas_treino = calcular_metricas(y_train, y_pred_train, y_proba_train)
        logar_metricas(metricas_treino, prefixo="train_")

        logger.info("  Métricas de teste:")
        metricas_teste = calcular_metricas(y_test, y_pred_test, y_proba_test)
        logar_metricas(metricas_teste, prefixo="test_")

        log_overfitting(metricas_treino, metricas_teste)
        calcular_custo_negocio(y_test, y_pred_test, "Random Forest")

        cm_path = salvar_matriz_confusao(y_test, y_pred_test, "Random Forest")
        fi_path = salvar_feature_importance(
            modelo.feature_importances_, feature_names, "Random Forest"
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))
        mlflow.sklearn.log_model(modelo, "model")

    logger.info("  Run MLflow registrado: Random Forest")


# ---------------------------------------------------------------------------
# Modelo 2 — Gradient Boosting
# ---------------------------------------------------------------------------

def treinar_gradient_boosting(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    info_dataset: dict,
) -> None:
    """
    Gradient Boosting sequencial.
    Cada nova árvore aprende os resíduos da anterior → reduz viés.
    learning_rate pequeno + n_estimators alto = melhor generalização.
    Conforme Fundamentos, Aula 04.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 5 — Gradient Boosting (Boosting sequencial)")
    logger.info("=" * 60)

    params = {
        "n_estimators":  300,
        "learning_rate": 0.05,   # shrinkage pequeno → melhor generalização
        "max_depth":     4,      # árvores rasas no boosting (evita overfitting)
        "subsample":     0.8,    # subamostragem → diversidade e menos variância
        "min_samples_leaf": 5,
        "random_state":  SEED,
    }

    with mlflow.start_run(run_name="gradient_boosting_300"):
        for k, v in info_dataset.items():
            mlflow.log_param(k, v)

        modelo = GradientBoostingClassifier(**params)
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
        calcular_custo_negocio(y_test, y_pred_test, "Gradient Boosting")

        cm_path = salvar_matriz_confusao(y_test, y_pred_test, "Gradient Boosting")
        fi_path = salvar_feature_importance(
            modelo.feature_importances_, feature_names, "Gradient Boosting"
        )

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))
        mlflow.sklearn.log_model(modelo, "model")

    logger.info("  Run MLflow registrado: Gradient Boosting")


# ---------------------------------------------------------------------------
# Tabela comparativa
# ---------------------------------------------------------------------------

def imprimir_tabela_comparativa() -> None:
    logger.info("\n%s", "=" * 60)
    logger.info("TABELA COMPARATIVA — Todos os Modelos")
    logger.info("=" * 60)

    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT],
        order_by=["metrics.test_auc_roc DESC"],
    )

    if runs.empty:
        logger.warning("Nenhum run encontrado.")
        return

    colunas = [
        "tags.mlflow.runName",
        "metrics.test_auc_roc",
        "metrics.test_recall",
        "metrics.test_f1",
        "metrics.test_precision",
        "metrics.negocio_resultado_liquido_BRL",
    ]
    cols = [c for c in colunas if c in runs.columns]
    tabela = runs[cols].copy()
    tabela.columns = [c.replace("metrics.", "").replace("tags.mlflow.", "") for c in cols]
    tabela = tabela.drop_duplicates(subset=["runName"])

    logger.info("\n%s", tabela.to_string(index=False))
    logger.info("\nMetas: AUC-ROC >= 0.80 | Recall >= 0.70")
    logger.info("Meta de negócio: Resultado líquido crescente a cada modelo.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Ensemble — Telco Customer Churn | seed=%d", SEED)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    info_dataset = obter_info_dataset(DATA_PROCESSED)
    logger.info("Dataset version: %s", info_dataset["dataset_version"])

    X_train, X_test, y_train, y_test, feature_names = preparar_dados(DATA_PROCESSED)

    treinar_random_forest(X_train, X_test, y_train, y_test, feature_names, info_dataset)
    treinar_gradient_boosting(X_train, X_test, y_train, y_test, feature_names, info_dataset)

    imprimir_tabela_comparativa()

    logger.info("\n03_ensemble.py concluído.")


if __name__ == "__main__":
    main()