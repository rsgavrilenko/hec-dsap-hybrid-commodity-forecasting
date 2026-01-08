import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ------------------
# Config
# ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "copper" / "data_copper_lme_all_years.csv"
FIG_PATH = Path(__file__).parent / "copper_price_stock_timeseries.png"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")
    return df


def plot_price_and_stock(df: pd.DataFrame):
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ---- Price (left axis)
    ax1.plot(
        df["date"],
        df["lme_copper_cash"],
        color="tab:blue",
        label="LME Copper Cash Price (USD)"
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # ---- Stock (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
        df["date"],
        df["lme_copper_stock"],
        color="tab:orange",
        alpha=0.6,
        label="LME Copper Stock"
    )
    ax2.set_ylabel("Stock (tons)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # ---- Title & legend
    fig.suptitle(
        "LME Copper Price and Stock Levels (2008–2025)",
        fontsize=14,
        fontweight="bold"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    plot_price_and_stock(df)
    print(f"Saved plot to {FIG_PATH}")
