"""
notebooks/04_mlp.py
MLP com PyTorch + Comparação Final — Telco Customer Churn

Baseado em Fundamentos, Aula 05 e Ciclo de Vida, Aula 02:
    - Arquitetura MLP: nn.Linear + ReLU + Dropout + Sigmoid
    - Loop de treino: forward → BCELoss → backprop → Adam step
    - Early stopping: interrompe se val_loss não melhora por `patience` épocas
    - Checkpoint: salva e restaura os melhores pesos
    - MLflow: parâmetros, métricas, dataset version, curva de loss
    - Tabela comparativa final: todos os modelos das Etapas 1 e 2

Uso:
    python notebooks/04_mlp.py
"""

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_features import preparar_dados
from src.models.mlp import ChurnMLP
from src.models.trainer import predizer, preparar_dataloaders, treinar_mlp
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_PROCESSED    = Path("data/processed/telco_clean.csv")
MLFLOW_EXPERIMENT = "telco_churn_baselines"
OUTPUT_DIR        = Path("outputs/mlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH   = Path("models/mlp_best.pt")

# Premissas de negócio
RECEITA_MENSAL_MEDIA      = 70.0
LTV_MESES                 = 12
CUSTO_CAMPANHA_RETENCAO   = 50.0
RECEITA_POR_CHURN_EVITADO = RECEITA_MENSAL_MEDIA * LTV_MESES
CUSTO_POR_FALSO_POSITIVO  = CUSTO_CAMPANHA_RETENCAO


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def obter_info_dataset(caminho: Path) -> dict:
    df       = pd.read_csv(caminho)
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


def calcular_custo_negocio(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    receita_preservada = tp * RECEITA_POR_CHURN_EVITADO
    custo_fp           = fp * CUSTO_POR_FALSO_POSITIVO
    receita_perdida    = fn * RECEITA_POR_CHURN_EVITADO
    resultado_liquido  = receita_preservada - custo_fp

    logger.info("\n  --- Custo de Negócio — MLP ---")
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


def salvar_curva_loss(historico: dict) -> Path:
    """Plota e salva a curva de loss treino vs validação por época."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(historico["train_loss"], label="Treino", color="#378ADD")
    ax.plot(historico["val_loss"],   label="Validação", color="#D85A30")
    ax.axvline(
        x=historico["epocas_treinadas"] - 1,
        linestyle="--", color="gray", alpha=0.6,
        label=f"Early stopping (época {historico['epocas_treinadas']})",
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("BCELoss")
    ax.set_title("Curva de Loss — MLP (Treino vs Validação)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    caminho = OUTPUT_DIR / "mlp_loss_curve.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("  Curva de loss salva: %s", caminho)
    return caminho


def salvar_matriz_confusao(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nome: str,
) -> Path:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Não churn", "Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Matriz de Confusão — {nome}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    slug    = nome.lower().replace(" ", "_")
    caminho = OUTPUT_DIR / f"confusion_matrix_{slug}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    logger.info("  Matriz de confusão salva: %s", caminho)
    return caminho


# ---------------------------------------------------------------------------
# Treino da MLP
# ---------------------------------------------------------------------------

def treinar_e_avaliar_mlp(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    info_dataset: dict,
) -> None:
    """
    Treina a MLP com early stopping e registra tudo no MLflow.

    Split interno: 80% treino / 20% validação (do X_train original).
    O X_test fica intocado para avaliação final.
    """
    logger.info("\n%s", "=" * 60)
    logger.info("MODELO 6 — MLP PyTorch (early stopping)")
    logger.info("=" * 60)

    # Split treino → treino + validação (para early stopping)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )
    logger.info(
        "Split treino/validação: treino=%d | val=%d | teste=%d",
        len(X_tr), len(X_val), len(X_test),
    )

    input_dim   = X_tr.shape[1]
    hidden_dims = [64, 32, 16]
    dropout     = 0.3
    lr          = 1e-3
    batch_size  = 64
    epochs      = 150
    patience    = 15

    params_mlp = {
        "input_dim":   input_dim,
        "hidden_dims": str(hidden_dims),
        "dropout":     dropout,
        "lr":          lr,
        "batch_size":  batch_size,
        "epochs":      epochs,
        "patience":    patience,
        "optimizer":   "Adam",
        "loss":        "BCELoss",
        "seed":        SEED,
    }

    with mlflow.start_run(run_name="mlp_pytorch_early_stopping"):
        # Dataset version
        for k, v in info_dataset.items():
            mlflow.log_param(k, v)

        # Parâmetros do modelo
        for k, v in params_mlp.items():
            mlflow.log_param(k, v)

        # Instanciar modelo
        modelo = ChurnMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        logger.info("  Arquitetura MLP:\n%s", modelo)

        # Preparar dataloaders
        train_loader, val_loader = preparar_dataloaders(
            X_tr, y_tr, X_val, y_val, batch_size=batch_size
        )

        # Treinar com early stopping
        historico = treinar_mlp(
            modelo=modelo,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            patience=patience,
            caminho_checkpoint=CHECKPOINT_PATH,
        )

        mlflow.log_metric("epocas_treinadas",  historico["epocas_treinadas"])
        mlflow.log_metric("melhor_val_loss",    historico["melhor_val_loss"])

        # Predições no conjunto de teste
        y_pred_train, y_proba_train = predizer(modelo, X_tr)
        y_pred_test,  y_proba_test  = predizer(modelo, X_test)

        logger.info("  Métricas de treino:")
        metricas_treino = calcular_metricas(y_tr, y_pred_train, y_proba_train)
        logar_metricas(metricas_treino, prefixo="train_")

        logger.info("  Métricas de teste:")
        metricas_teste = calcular_metricas(y_test, y_pred_test, y_proba_test)
        logar_metricas(metricas_teste, prefixo="test_")

        # Gap de overfitting
        gap = metricas_treino["accuracy"] - metricas_teste["accuracy"]
        mlflow.log_metric("overfitting_gap_accuracy", gap)
        logger.info("  %-35s %.4f", "overfitting_gap_accuracy", gap)

        # Custo de negócio
        calcular_custo_negocio(y_test, y_pred_test)

        # Artefatos
        loss_path = salvar_curva_loss(historico)
        cm_path   = salvar_matriz_confusao(y_test, y_pred_test, "MLP PyTorch")

        mlflow.log_artifact(str(loss_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(CHECKPOINT_PATH))

    logger.info("  Run MLflow registrado: MLP PyTorch")


# ---------------------------------------------------------------------------
# Tabela comparativa final (todos os modelos)
# ---------------------------------------------------------------------------

def imprimir_tabela_final() -> None:
    logger.info("\n%s", "=" * 60)
    logger.info("TABELA COMPARATIVA FINAL — Etapas 1 e 2")
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
        "metrics.overfitting_gap_accuracy",
        "metrics.negocio_resultado_liquido_BRL",
    ]
    cols   = [c for c in colunas if c in runs.columns]
    tabela = runs[cols].copy()
    tabela.columns = [
        c.replace("metrics.", "").replace("tags.mlflow.", "") for c in cols
    ]
    tabela = tabela.drop_duplicates(subset=["runName"])

    logger.info("\n%s", tabela.to_string(index=False))
    logger.info("\n%s", "-" * 60)
    logger.info("Metas técnicas: AUC-ROC >= 0.80 | Recall >= 0.70")
    logger.info("Meta de negócio: Resultado líquido maior que o baseline.")
    logger.info(
        "Próximo passo: refatorar melhor modelo em src/ e servir via FastAPI."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("MLP PyTorch — Telco Customer Churn | seed=%d", SEED)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    info_dataset = obter_info_dataset(DATA_PROCESSED)
    logger.info("Dataset version: %s", info_dataset["dataset_version"])

    X_train, X_test, y_train, y_test, feature_names = preparar_dados(DATA_PROCESSED)
    logger.info("Input dim: %d features", X_train.shape[1])

    treinar_e_avaliar_mlp(X_train, X_test, y_train, y_test, info_dataset)

    imprimir_tabela_final()

    logger.info("\nEtapa 2 concluída.")


if __name__ == "__main__":
    main()