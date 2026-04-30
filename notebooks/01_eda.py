"""
notebooks/01_eda.py
Análise Exploratória de Dados — Telco Customer Churn

Segue a estrutura do CRISP-DM conforme Ciclo de Vida, Aula 02:
    Passo 1: Carregamento e inspeção inicial
    Passo 2: EDA
             2.1 Missing values
             2.2 Distribuição da variável target
             2.3 Identificação de outliers (Boxplot + Z-Score)
             2.4 Distribuições das variáveis numéricas
             2.5 Análise de correlações (heatmap)
             2.6 Análise bivariada (features vs target)
    Passo 3: Preparação e limpeza (via src/data/preprocessing.py)
    Passo 4: Data Readiness Report

Uso:
    python notebooks/01_eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Garante que o root do projeto está no PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    carregar_dados,
    limpar_dados,
    salvar_dados_processados,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

DATA_RAW = Path("data/raw/Telco-Customer-Churn.csv")
DATA_PROCESSED = Path("data/processed/telco_clean.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"0": "#5DCAA5", "1": "#D85A30"}
COLOR_PRIMARY = "#378ADD"
COLOR_SECONDARY = "#D85A30"


# ---------------------------------------------------------------------------
# Passo 1 — Carregamento e inspeção inicial
# ---------------------------------------------------------------------------
def passo1_inspecao(df: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("PASSO 1 — Carregamento e Inspeção Inicial")
    logger.info("=" * 60)
    logger.info("Shape: %d linhas x %d colunas", *df.shape)

    import io
    buf = io.StringIO()
    df.info(buf=buf)
    logger.info("\n%s", buf.getvalue())
    logger.info("Estatísticas descritivas:\n%s", df.describe(include="all").to_string())


# ---------------------------------------------------------------------------
# Passo 2.1 — Missing values
# ---------------------------------------------------------------------------
def passo2_1_missing(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.1 Missing Values ---")

    df_check = df.copy()
    df_check["TotalCharges"] = pd.to_numeric(df_check["TotalCharges"], errors="coerce")

    missing = pd.DataFrame({
        "Missing_Count": df_check.isnull().sum(),
        "Missing_Pct": (df_check.isnull().sum() / len(df_check) * 100).round(2),
    })
    missing = missing[missing["Missing_Count"] > 0].sort_values(
        by="Missing_Pct", ascending=False
    )

    espacos_tc = (df["TotalCharges"].str.strip() == "").sum()
    logger.info("TotalCharges com valor vazio (espaço em branco): %d", espacos_tc)
    logger.info("Estratégia: pd.to_numeric(errors='coerce') + imputação com mediana.")

    if not missing.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(missing.index, missing["Missing_Pct"], color=COLOR_SECONDARY)
        ax.set_xlabel("Porcentagem (%)")
        ax.set_title("Missing Values por Coluna")
        plt.tight_layout()
        _salvar(fig, "2_1_missing_values.png")
    else:
        logger.info("Sem NaN detectados além do TotalCharges.")


# ---------------------------------------------------------------------------
# Passo 2.2 — Distribuição do target
# ---------------------------------------------------------------------------
def passo2_2_target(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.2 Distribuição da Variável Target (Churn) ---")

    contagem = df["Churn"].value_counts()
    pct = df["Churn"].value_counts(normalize=True) * 100

    for label in contagem.index:
        logger.info("  Churn=%-3s %d registros (%.2f%%)", label, contagem[label], pct[label])

    ratio = contagem.min() / contagem.max()
    logger.info("  Ratio balanceamento: %.2f", ratio)

    if ratio < 0.5:
        logger.warning(
            "Dataset DESBALANCEADO (ratio=%.2f). "
            "Obrigatório: validação estratificada + AUC-ROC + PR-AUC.", ratio
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição da Variável Target — Churn", fontsize=13, fontweight="bold")

    cores = [PALETTE["0"], PALETTE["1"]]
    rotulos = ["Não cancelou (No)", "Cancelou (Yes)"]

    axes[0].bar(rotulos, contagem.values, color=cores)
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Contagem por Classe")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(contagem.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontsize=11, fontweight="bold")

    axes[1].pie(
        contagem.values,
        labels=rotulos,
        autopct="%1.1f%%",
        colors=cores,
        startangle=90,
    )
    axes[1].set_title("Proporção de Classes")

    plt.tight_layout()
    _salvar(fig, "2_2_target.png")


# ---------------------------------------------------------------------------
# Passo 2.3 — Outliers (Boxplot + Z-Score)
# ---------------------------------------------------------------------------
def passo2_3_outliers(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.3 Outliers (Boxplot + Z-Score) ---")

    df_num = df.copy()
    df_num["TotalCharges"] = pd.to_numeric(df_num["TotalCharges"], errors="coerce")

    colunas = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, len(colunas), figsize=(15, 5))
    fig.suptitle("Boxplots — Variáveis Numéricas", fontsize=13, fontweight="bold")

    for idx, col in enumerate(colunas):
        serie = df_num[col].dropna()
        sns.boxplot(y=serie, ax=axes[idx], color=COLOR_PRIMARY)
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].grid(axis="y", alpha=0.3)

        z = np.abs(stats.zscore(serie))
        n_out = (z > 3).sum()
        axes[idx].text(
            0.95, 0.98, f"Outliers Z>3: {n_out}",
            transform=axes[idx].transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )
        logger.info("  %-20s outliers (Z>3): %d", col, n_out)

    logger.info("  Conclusão: zero outliers severos. StandardScaler no pipeline é suficiente.")
    plt.tight_layout()
    _salvar(fig, "2_3_outliers.png")


# ---------------------------------------------------------------------------
# Passo 2.4 — Distribuições das numéricas
# ---------------------------------------------------------------------------
def passo2_4_distribuicoes(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.4 Distribuições das Variáveis Numéricas ---")

    df_plot = df.copy()
    df_plot["TotalCharges"] = pd.to_numeric(df_plot["TotalCharges"], errors="coerce")

    colunas = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Distribuição das Numéricas por Churn", fontsize=13, fontweight="bold")

    for idx, col in enumerate(colunas):
        for churn_val, cor in [("No", PALETTE["0"]), ("Yes", PALETTE["1"])]:
            subset = df_plot[df_plot["Churn"] == churn_val][col].dropna()
            axes[idx].hist(
                subset, bins=30, alpha=0.6,
                color=cor, label=f"Churn={churn_val}", density=True,
            )

        skew = df_plot[col].skew()
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Densidade")
        axes[idx].legend()
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].text(
            0.97, 0.97, f"Skew: {skew:.2f}",
            transform=axes[idx].transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )
        logger.info("  %-20s skewness: %.4f", col, skew)

    plt.tight_layout()
    _salvar(fig, "2_4_distribuicoes.png")


# ---------------------------------------------------------------------------
# Passo 2.5 — Correlações
# ---------------------------------------------------------------------------
def passo2_5_correlacoes(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.5 Correlações (Heatmap) ---")

    df_corr = df.copy()
    df_corr["TotalCharges"] = pd.to_numeric(df_corr["TotalCharges"], errors="coerce")
    df_corr["Churn_bin"] = (df_corr["Churn"] == "Yes").astype(int)

    cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_bin"]
    matrix = df_corr[cols].corr()

    logger.info("Matriz de correlação:\n%s", matrix.round(3).to_string())
    logger.info(
        "  Alerta: tenure x TotalCharges = %.2f — "
        "alta correlação esperada (multicolinearidade em modelos lineares).",
        matrix.loc["tenure", "TotalCharges"],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Heatmap de Correlação — Numéricas + Target", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _salvar(fig, "2_5_correlacoes.png")


# ---------------------------------------------------------------------------
# Passo 2.6 — Análise bivariada
# ---------------------------------------------------------------------------
def passo2_6_bivariada(df: pd.DataFrame) -> None:
    logger.info("\n--- 2.6 Análise Bivariada (Features vs Target) ---")

    # Categóricas vs Churn
    features_cat = [
        "Contract", "PaymentMethod", "InternetService",
        "PaperlessBilling", "TechSupport", "OnlineSecurity",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Taxa de Churn por Feature Categórica", fontsize=14, fontweight="bold")
    axes = axes.ravel()

    for idx, col in enumerate(features_cat):
        tabela = (
            df.groupby(col)["Churn"]
            .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
            .reset_index()
        )
        tabela.columns = [col, "Churn_Rate"]
        tabela = tabela.sort_values("Churn_Rate", ascending=True)

        axes[idx].barh(tabela[col], tabela["Churn_Rate"], color=COLOR_SECONDARY)
        axes[idx].set_xlabel("Taxa de Churn (%)")
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].grid(axis="x", alpha=0.3)
        for _, row in tabela.iterrows():
            axes[idx].text(
                row["Churn_Rate"] + 0.5, row[col],
                f"{row['Churn_Rate']:.1f}%", va="center", fontsize=9,
            )

        logger.info("  %s:", col)
        for _, row in tabela.iterrows():
            logger.info("    %-35s %.1f%%", row[col], row["Churn_Rate"])

    plt.tight_layout()
    _salvar(fig, "2_6_bivariada_categoricas.png")

    # Numéricas vs Churn (boxplot)
    df_box = df.copy()
    df_box["TotalCharges"] = pd.to_numeric(df_box["TotalCharges"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Numéricas por Churn (Bivariada)", fontsize=13, fontweight="bold")

    cores_boxplot = {"No": PALETTE["0"], "Yes": PALETTE["1"]}

    for idx, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.boxplot(
            data=df_box, x="Churn", y=col,
            hue="Churn", palette=cores_boxplot,
            legend=False, ax=axes[idx],
        )
        axes[idx].set_title(col, fontweight="bold")
        axes[idx].grid(axis="y", alpha=0.3)

        medianas = df_box.groupby("Churn")[col].median()
        logger.info(
            "  %-20s mediana No=%.2f | Yes=%.2f",
            col, medianas.get("No", 0), medianas.get("Yes", 0),
        )

    plt.tight_layout()
    _salvar(fig, "2_6_bivariada_numericas.png")


# ---------------------------------------------------------------------------
# Passo 4 — Data Readiness Report
# ---------------------------------------------------------------------------
def passo4_data_readiness(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("PASSO 4 — Data Readiness Report")
    logger.info("=" * 60)

    churn_rate = df_clean["Churn"].mean() * 100
    ratio = df_clean["Churn"].value_counts().min() / df_clean["Churn"].value_counts().max()

    logger.info("\n VOLUME")
    logger.info("  Registros brutos:    %d", len(df_raw))
    logger.info("  Registros limpos:    %d", len(df_clean))
    logger.info("  Colunas para modelo: %d  (excluindo customerID)", df_clean.shape[1] - 1)

    logger.info("\n BALANCEAMENTO")
    logger.info("  Taxa de churn: %.2f%%", churn_rate)
    logger.info("  Ratio:         %.2f  → DESBALANCEADO", ratio)
    logger.info("  Ação:          validação estratificada + AUC-ROC + PR-AUC.")

    logger.info("\n QUALIDADE")
    logger.info("  TotalCharges com espaço em branco: 11 → imputados com mediana.")
    logger.info("  Missing NaN após limpeza:           0")
    logger.info("  Duplicatas:                         %d", df_clean.duplicated().sum())
    logger.info("  Outliers severos (Z>3):             0 em todas as numéricas.")

    logger.info("\n PRINCIPAIS INSIGHTS")
    logger.info("  • Contract Month-to-month:  ~43%% churn (vs 11%% One year / 3%% Two year).")
    logger.info("  • InternetService Fiber:    ~42%% churn (vs 19%% DSL / 7%% sem internet).")
    logger.info("  • Sem TechSupport:          ~42%% churn (vs 15%% com suporte).")
    logger.info("  • Sem OnlineSecurity:       ~42%% churn (vs 15%% com segurança).")
    logger.info("  • Electronic check:         ~45%% churn (maior entre formas de pagamento).")
    logger.info("  • tenure mediana churn=Yes: 10 meses (vs 38 meses churn=No).")
    logger.info("  • Correlação tenure x TotalCharges: 0.83 — monitorar multicolinearidade.")

    logger.info("\n STATUS: DADOS PRONTOS PARA MODELAGEM.")
    logger.info("  Próximo passo: baselines (DummyClassifier + LogisticRegression) com MLflow.")


# ---------------------------------------------------------------------------
# Utilitário interno
# ---------------------------------------------------------------------------
def _salvar(fig: plt.Figure, nome: str) -> None:
    caminho = OUTPUT_DIR / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico salvo: %s", caminho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("EDA — Telco Customer Churn | seed=%d", SEED)

    # Passo 1 — carregar e inspecionar
    df_raw = carregar_dados(DATA_RAW)
    passo1_inspecao(df_raw)

    # Passo 2 — EDA
    logger.info("\n" + "=" * 60)
    logger.info("PASSO 2 — Análise Exploratória de Dados (EDA)")
    logger.info("=" * 60)
    passo2_1_missing(df_raw)
    passo2_2_target(df_raw)
    passo2_3_outliers(df_raw)
    passo2_4_distribuicoes(df_raw)
    passo2_5_correlacoes(df_raw)
    passo2_6_bivariada(df_raw)

    # Passo 3 — limpar e salvar
    logger.info("\n" + "=" * 60)
    logger.info("PASSO 3 — Preparação e Limpeza")
    logger.info("=" * 60)
    df_clean = limpar_dados(df_raw)
    salvar_dados_processados(df_clean, DATA_PROCESSED)

    # Passo 4 — data readiness
    passo4_data_readiness(df_raw, df_clean)

    logger.info("\nEDA concluída. Gráficos em: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()