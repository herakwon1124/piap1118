"""Build the three-page PDF from the numbers produced by analysis.py."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

GITHUB_URL = "https://github.com/herakwon1124/piap1118"
STUDENT_NAME = "Hyeyun Kwon"
PAGE = (13.333, 7.5)


def _plain(text: str) -> str:
    return text.replace("$", r"\$")


def _money(value: float, cents: bool = False) -> str:
    sign = "−" if value < 0 else ""
    if cents:
        return f"{sign}${abs(value):,.2f}"
    return f"{sign}${abs(value):,.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_slides(facts: dict, pdf_path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    preview = pdf_path.parent / "_preview"
    preview.mkdir(exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        _title(pdf, preview)
        _fit_slide(pdf, facts, preview)
        _decision_slide(pdf, facts, preview)


def _save(fig, pdf, preview: Path, name: str) -> None:
    pdf.savefig(fig)
    fig.savefig(preview / name, dpi=130)
    plt.close(fig)


def _title(pdf, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(0.08, 0.64, "Modeling — decision tree", fontsize=18, color="0.35")
    fig.text(
        0.08,
        0.46,
        "Clinic no-shows:\nshould the clinic overbook this slot?",
        fontsize=28,
        va="top",
        linespacing=1.4,
    )
    fig.text(0.08, 0.24, STUDENT_NAME, fontsize=16)
    fig.text(0.08, 0.17, GITHUB_URL, fontsize=14)
    _save(fig, pdf, preview, "01_title.png")


def _fit_slide(pdf, facts: dict, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(
        0.05,
        0.93,
        "Overbook when the patient has missed before and got no text.",
        fontsize=18,
        va="top",
    )
    data = (
        "Data\n"
        f"Source: clinic_appointments.csv. {facts['n']} past appointments.\n"
        f"{facts['n_no_show']} were no-shows ({_pct(facts['no_show_rate'])}).\n"
        f"Split: stratified 70/30, seed {facts['random_state']}.\n"
        f"{facts['n_train']} to fit, {facts['n_test']} held out.\n"
        f"No-show rate stays {_pct(facts['train_no_show_rate'])} and "
        f"{_pct(facts['test_no_show_rate'])}."
    )
    model = (
        "Model\n"
        "Outcome: no-show or checked-in.\n"
        "Family: classification tree. Depth at most 3.\n"
        "A leaf needs at least 10 training appointments.\n"
        "A split is kept only if impurity falls by 0.01.\n"
        "All seven features were available.\n"
        "It splits on prior no-shows, then the text."
    )
    fig.text(0.05, 0.86, data, fontsize=12, va="top", linespacing=1.35)
    fig.text(0.05, 0.52, model, fontsize=12, va="top", linespacing=1.35)

    by_count = {(leaf["n"], leaf["no_shows"]): leaf for leaf in facts["leaves"]}

    def branch(n: int, misses: int, label: str) -> str:
        leaf = by_count[(n, misses)]
        return f"{label}\n    {leaf['decision']} — {leaf['no_shows']} of {leaf['n']} missed"

    tree = (
        "The fitted tree\n"
        + branch(17, 15, "3 or more prior no-shows")
        + "\n"
        + branch(155, 10, "Fewer than 3, text sent")
        + "\n"
        + branch(56, 10, "No text, no prior miss")
        + "\n"
        + branch(52, 29, "1 or 2 prior misses, no text")
    )
    fig.text(0.58, 0.86, tree, fontsize=12, va="top", linespacing=1.3)

    base = facts["baseline"]
    test = facts["test"]
    fit = (
        "Baseline vs v1, on the 120 appointments the tree never saw\n"
        f"Hold every slot: accuracy {_pct(base['accuracy'])}. "
        f"It predicts no no-shows, so precision is not defined and recall is 0.\n"
        f"Confusion: {base['tn']} held shows, {base['fp']} false overbooks, "
        f"{base['fn']} missed no-shows, {base['tp']} caught no-shows.\n"
        f"Tree: accuracy {_pct(test['accuracy'])}. "
        f"No-show precision {_pct(test['precision'])}, recall {_pct(test['recall'])}.\n"
        f"Confusion: {test['tn']} held shows, {test['fp']} false overbooks, "
        f"{test['fn']} missed no-shows, {test['tp']} caught no-shows.\n"
        f"Training accuracy {_pct(facts['train_accuracy'])} next to test "
        f"{_pct(test['accuracy'])}. Training is not higher, so the tree did not memorize."
    )
    fig.text(0.05, 0.28, fit, fontsize=11.5, va="top", linespacing=1.35)
    _save(fig, pdf, preview, "02_fit.png")


def _decision_slide(pdf, facts: dict, preview: Path) -> None:
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(
        0.05,
        0.92,
        _plain(
            f"Overbook those slots. Worth {_money(facts['per_gap'], cents=True)} "
            f"per appointment, {_money(facts['annual_value'])} a year."
        ),
        fontsize=18,
        va="top",
    )
    fig.text(
        0.05,
        0.84,
        "Rule: overbook if the tree predicts a no-show. Hold the slot if it predicts a show.",
        fontsize=13,
        va="top",
    )

    d = facts["D"]
    e = facts["E"]
    cases = (
        "D — booked 21 days ahead, 1 prior miss, age 27, 9.5 miles, no text, "
        "first visit, 3pm.\n"
        f"Leaf: 1 or 2 prior misses and no text. "
        f"{d['no_shows']} of {d['n']} training visits in that leaf were misses. "
        f"{d['decision'].capitalize()}.\n\n"
        "E — booked 3 days ahead, no prior miss, age 61, 2 miles, text sent, "
        "returning patient, 9am.\n"
        f"Leaf: a text was sent and fewer than 3 prior misses. "
        f"{e['no_shows']} of {e['n']} training visits in that leaf were misses. "
        f"{e['decision'].capitalize()}."
    )
    fig.text(0.05, 0.76, cases, fontsize=12.5, va="top", linespacing=1.4)

    base = facts["baseline"]
    test = facts["test"]
    cells = [
        ["Policy", "Value on the 120 test visits", "Per appointment"],
        [
            "Hold every slot",
            _money(base["value"]),
            _money(base["value_per"]),
        ],
        [
            "Tree",
            _money(test["value"]),
            _money(test["value_per"], cents=True),
        ],
    ]
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.24])
    ax.axis("off")
    table = ax.table(cellText=cells, loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.0)
    for (r, _c), cell in table.get_celld().items():
        cell.set_width(0.33)
        cell.set_edgecolor("0.7")
        cell.set_linewidth(0.5)
        cell.set_facecolor("0.95" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(fontweight="bold")

    note = (
        "An empty slot costs $120. Overbooking a patient who shows up costs $60. "
        "A correct call costs $0.\n"
        f"Yearly figure assumes {facts['annual_appointments']:,} appointments: "
        "20 a day, 5 days a week, 50 weeks. "
        f"{_money(facts['per_gap'], cents=True)} times that volume is "
        f"{_money(facts['annual_value'])}."
    )
    fig.text(0.05, 0.24, _plain(note), fontsize=12.5, va="top", linespacing=1.45)
    _save(fig, pdf, preview, "03_decision.png")
