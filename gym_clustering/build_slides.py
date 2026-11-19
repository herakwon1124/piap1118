"""Build the PDF from the numbers produced by analysis.py.

Pages: title, then the two slides for the operations lead.
Black text, white background. The chart is drawn from the same WCSS table
the script prints.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Title-slide link. This is the repo the code is meant to live in.
# The machine this was written on could not authenticate to GitHub.
GITHUB_URL = "https://github.com/herakwon1124/piap1118"

STUDENT_NAME = "Hyeyun Kwon"
PAGE = (13.333, 7.5)


def _plain(text: str) -> str:
    """Matplotlib treats $...$ as math. Keep dollar signs as text."""
    return text.replace("$", r"\$")


def _money(value: float) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _whole(value: float) -> str:
    return f"{value:,.0f}"


def write_slides(facts: dict, spread: pd.DataFrame, pdf_path: Path, elbow_png: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    preview = pdf_path.parent / "_preview"
    preview.mkdir(exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _title(pdf, preview)
        _fit_slide(pdf, facts, spread, preview)
        _decision_slide(pdf, facts, preview)
    _ = elbow_png


def _save(fig, pdf: PdfPages, preview: Path, name: str) -> None:
    pdf.savefig(fig)
    fig.savefig(preview / name, dpi=130)
    plt.close(fig)


def _title(pdf: PdfPages, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(0.08, 0.64, _plain("Modeling — clustering"), fontsize=18, color="0.35")
    fig.text(
        0.08,
        0.46,
        _plain("Gym member segments:\nwho should get the monthly offer?"),
        fontsize=28,
        color="black",
        va="top",
        linespacing=1.4,
    )
    fig.text(0.08, 0.24, _plain(STUDENT_NAME), fontsize=16, color="black")
    fig.text(0.08, 0.17, _plain(GITHUB_URL), fontsize=14, color="black")
    _save(fig, pdf, preview, "01_title.png")


def _fit_slide(pdf: PdfPages, facts: dict, spread: pd.DataFrame, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(
        0.05,
        0.93,
        _plain("Four groups fit. A fifth group barely moves the spread."),
        fontsize=20,
        color="black",
        va="top",
    )

    data = (
        "Data\n"
        f"Source: gym_members.csv. {facts['n_members']} members, one row each, "
        f"{facts['n_columns']} columns.\n"
        "Period: averages over the last 3 months. "
        f"Mean extras spend is ${facts['mean_extras']:.0f} a month.\n"
        f"Split: random 70/30, seed {facts['random_state']}. "
        f"{facts['n_train']} to fit, {facts['n_test']} held out.\n"
        "Scaler fit on the training members only.\n"
        "Test members are not used to fit."
    )
    model = (
        "Model\n"
        "Family: k-means. k = 4.\n"
        "Features: age, visits per month, minutes per visit, classes per month,\n"
        "months as a member, monthly extras spend.\n"
        "Left out: member_id (an ID, not how someone uses the gym).\n"
        "Scaling: each feature set to mean 0 and sd 1, using the training members."
    )
    fig.text(0.05, 0.84, _plain(data), fontsize=12, color="black", va="top", linespacing=1.4)
    fig.text(0.05, 0.60, _plain(model), fontsize=12, color="black", va="top", linespacing=1.4)

    ax = fig.add_axes([0.64, 0.38, 0.32, 0.44])
    ax.plot(spread["k"], spread["train_wcss"], marker="o", color="black", label="Training")
    ax.plot(
        spread["k"],
        spread["test_wcss"],
        marker="o",
        color="0.4",
        linestyle="--",
        label="Test",
    )
    ax.axvline(facts["chosen_k"], color="0.75", linewidth=1)
    ax.set_xlabel("Number of groups (k)", fontsize=10)
    ax.set_ylabel("Within-group spread", fontsize=10)
    ax.set_xticks(list(spread["k"]))
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Spread vs k", fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fit = (
        "Baseline vs this model, on the 99 test members\n"
        f"Spread at k = 1: {_whole(facts['test_wcss_k1'])}.    "
        f"Spread at k = 4: {_whole(facts['test_wcss_k'])}.\n"
        "Why k = 4: the drop is still large through 4 groups, then it flattens. "
        f"From 3 to 4, test spread falls by {facts['test_drop_3_to_4']:.0f}. "
        f"From 4 to 5, it falls by {facts['test_drop_4_to_5']:.0f}.\n"
        "The test curve bends at the same k, so the groups are not just a training-set pattern."
    )
    fig.text(0.05, 0.28, _plain(fit), fontsize=12, color="black", va="top", linespacing=1.45)
    _save(fig, pdf, preview, "02_fit.png")


def _decision_slide(pdf: PdfPages, facts: dict, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    lift = facts["v1_net"] - facts["baseline_net"]
    fig.text(
        0.05,
        0.93,
        _plain(
            f"Send the $10 credit only to newcomers. Worth {_money(lift)} a month."
        ),
        fontsize=20,
        color="black",
        va="top",
    )
    fig.text(
        0.05,
        0.855,
        _plain(
            "Compared with sending it to every member: "
            f"{_money(facts['baseline_net'])} a month today, "
            f"{_money(facts['v1_net'])} a month if only newcomers get it."
        ),
        fontsize=12.5,
        color="black",
        va="top",
    )

    order = ["Newcomers", "Veterans", "Floor regulars", "Studio members"]
    by_name = {row["segment_name"]: row for row in facts["segments"]}
    cells = [["Segment", "Members", "Defining traits", "Extras / month", "Credit"]]
    nets = []
    for name in order:
        row = by_name[name]
        cells.append(
            [
                name,
                f"{row['n']}",
                row["traits"],
                f"${row['extras']:.0f}",
                "Yes" if row["gets_credit"] else "No",
            ]
        )
        nets.append((name, row["net_if_credited"], row["gets_credit"]))

    ax = fig.add_axes([0.05, 0.55, 0.90, 0.25])
    ax.axis("off")
    table = ax.table(cellText=cells, loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.85)
    widths = [0.16, 0.11, 0.48, 0.14, 0.11]
    for (r, c), cell_obj in table.get_celld().items():
        cell_obj.set_width(widths[c])
        cell_obj.set_edgecolor("0.7")
        cell_obj.set_linewidth(0.5)
        cell_obj.set_facecolor("white")
        if r == 0:
            cell_obj.set_text_props(fontweight="bold")
            cell_obj.set_facecolor("0.95")

    new = facts["new_member"]
    gets = "gets the credit" if new["gets_credit"] else "does not get the credit"
    veteran_net = by_name["Veterans"]["net_if_credited"]
    floor_net = by_name["Floor regulars"]["net_if_credited"]
    studio_net = by_name["Studio members"]["net_if_credited"]
    footer = (
        f"New member (age 30, 4 visits, 42 minutes, 1 class, 5 months, $10 extras) "
        f"is a {new['segment_name'].rstrip('s')} and {gets}.\n"
        "Scored member by member, not off the segment average. "
        "Under $20 of extras and they get the credit: +$5. "
        "At $20 or more: −$10. No credit: $0.\n"
        "Traits above are the training centers. "
        f"Veterans center at ${by_name['Veterans']['extras']:.0f}, still under $20, "
        f"but crediting every veteran nets {_money(veteran_net)}.\n"
        f"Floor regulars {_money(floor_net)}. Studio members {_money(studio_net)}. "
        "Those three do not get the credit."
    )
    fig.text(
        0.05,
        0.50,
        _plain(footer),
        fontsize=12,
        color="black",
        va="top",
        linespacing=1.45,
    )
    _save(fig, pdf, preview, "03_decision.png")
