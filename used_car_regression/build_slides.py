"""Build the three-page PDF from the numbers produced by analysis.py.

Title page, then the two slides for the dealer's owner.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

GITHUB_URL = "https://github.com/herakwon1124/piap1118"
STUDENT_NAME = "Hyeyun Kwon"
PAGE = (13.333, 7.5)


def _plain(text: str) -> str:
    return text.replace("$", r"\$")


def _money(value: float) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def write_slides(facts: dict, pdf_path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    preview = pdf_path.parent / "_preview"
    preview.mkdir(exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _title(pdf, preview)
        _fit_slide(pdf, facts, preview)
        _decision_slide(pdf, facts, preview)


def _save(fig, pdf: PdfPages, preview: Path, name: str) -> None:
    pdf.savefig(fig)
    fig.savefig(preview / name, dpi=130)
    plt.close(fig)


def _title(pdf: PdfPages, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(0.08, 0.64, "Modeling — regression", fontsize=18, color="0.35")
    fig.text(
        0.08,
        0.46,
        "Used car prices:\nshould the dealer buy this car?",
        fontsize=28,
        color="black",
        va="top",
        linespacing=1.4,
    )
    fig.text(0.08, 0.24, STUDENT_NAME, fontsize=16)
    fig.text(0.08, 0.17, GITHUB_URL, fontsize=14)
    _save(fig, pdf, preview, "01_title.png")


def _fit_slide(pdf: PdfPages, facts: dict, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(
        0.05,
        0.93,
        "Four features price a car. A fifth one does not.",
        fontsize=20,
        va="top",
    )

    coef = {row["name"]: row for row in facts["coefs"]}
    data = (
        "Data\n"
        f"Source: used_car_sales.csv. {facts['n_sales']} cars the dealer sold "
        "in the last year.\n"
        f"Split: random 70/30, seed {facts['random_state']}. "
        f"{facts['n_train']} cars to fit, {facts['n_test']} held out.\n"
        f"The {facts['n_lots']} auction lots are not used to fit the model."
    )
    model = (
        "Model\n"
        "Outcome: sale price on the lot. Family: linear regression.\n"
        "v1 features: mileage (in thousands), age, horsepower, accident.\n"
        f"Dropped: number of previous owners (p = {facts['owners_pvalue']:.2f}).\n"
        f"Adjusted R² is {facts['full_adj_r2']:.3f} with it "
        f"and {facts['v1_adj_r2']:.3f} without it.\n"
        "The simpler model is the one to use.\n"
        f"Training adjusted R² for v1: {facts['v1_adj_r2']:.3f}.\n"
        "Each of the four p-values is under 0.001.\n"
        f"Per 1,000 miles: {_money(coef['Mileage_K']['coef'])}.\n"
        f"Per year of age: {_money(coef['Age_Years']['coef'])}.\n"
        f"Per horsepower: +{_money(coef['Horsepower']['coef'])}.\n"
        f"An accident on the history: {_money(coef['Accident_yes']['coef'])}."
    )
    fig.text(0.05, 0.86, _plain(data), fontsize=12, va="top", linespacing=1.4)
    fig.text(0.05, 0.66, _plain(model), fontsize=12, va="top", linespacing=1.38)

    actual = np.asarray(facts["actual"])
    predicted = np.asarray(facts["predicted"])
    ax = fig.add_axes([0.66, 0.34, 0.29, 0.48])
    ax.scatter(actual, predicted, s=18, color="black", zorder=3)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    pad = (high - low) * 0.05
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="0.55", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Actual sale price", fontsize=10)
    ax.set_ylabel("Predicted sale price", fontsize=10)
    ax.set_title("72 test cars", fontsize=12)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v/1000:.0f}k"))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fit = (
        "Baseline vs v1, on the 72 cars the model never saw\n"
        f"One average price (${facts['train_mean']:,.0f}, the training mean): "
        f"R² {facts['test_r2_base']:.3f}, typical error {_money(facts['test_rmse_base'])}.\n"
        f"v1: R² {facts['test_r2_v1']:.3f}, typical error {_money(facts['test_rmse_v1'])}. "
        "The line on the chart is a perfect prediction."
    )
    fig.text(0.05, 0.26, _plain(fit), fontsize=12, va="top", linespacing=1.45)
    _save(fig, pdf, preview, "02_fit.png")


def _decision_slide(pdf: PdfPages, facts: dict, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    base = facts["baseline_auction"]
    v1 = facts["v1_auction"]
    lift = v1["profit"] - base["profit"]
    fig.text(
        0.05,
        0.92,
        _plain(
            f"Bid with the model. On these 40 cars it is worth {_money(lift)} more."
        ),
        fontsize=20,
        va="top",
    )
    fig.text(
        0.05,
        0.84,
        _plain(
            "Buy a car when its predicted sale price is higher than the auction "
            f"price plus {_money(facts['prep_cost'])} to put it on the lot. Otherwise pass."
        ),
        fontsize=13,
        va="top",
    )

    cells = [
        ["Rule", "Cars bought", "Of those, lose money", "Total profit"],
        [
            "Average price",
            str(base["bought"]),
            str(base["losers"]),
            _money(base["profit"]),
        ],
        [
            "v1, the regression",
            str(v1["bought"]),
            str(v1["losers"]),
            _money(v1["profit"]),
        ],
    ]
    ax = fig.add_axes([0.08, 0.42, 0.84, 0.32])
    ax.axis("off")
    table = ax.table(cellText=cells, loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.1)
    for (r, c), cell in table.get_celld().items():
        cell.set_width(0.25)
        cell.set_edgecolor("0.7")
        cell.set_linewidth(0.5)
        cell.set_facecolor("0.95" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(fontweight="bold")

    note = (
        f"The average-price rule buys {base['bought']} cars and "
        f"{base['losers']} of them lose money, for {_money(base['profit'])}.\n"
        f"The model buys {v1['bought']} cars. {v1['losers']} of those still lose money, "
        f"but the wins cover them: {_money(v1['profit'])} in total.\n"
        "Profit on a car the rule buys is the later retail price, minus the auction "
        f"price, minus {_money(facts['prep_cost'])}. Cars the rule passes are not in the total."
    )
    fig.text(0.08, 0.38, _plain(note), fontsize=13, va="top", linespacing=1.5)
    _save(fig, pdf, preview, "03_decision.png")
