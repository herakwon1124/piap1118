"""Clinic no-show decision tree.

One command reproduces the fit, the two appointment decisions,
the test-set value, and the slides:

    python3 analysis.py

Split: stratified 70/30, random_state=42.
The tree is fit on the training appointments only.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from build_slides import write_slides

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "clinic_appointments.csv"

FEATURES = [
    "lead_time_days",
    "prior_no_shows",
    "age",
    "distance_miles",
    "sms_reminder",
    "new_patient",
    "appointment_hour",
]

# Depth 3 is enough. A leaf must hold 10 training appointments.
# An impurity drop of 0.01 drops a split that does not change the call
# (an age cut where both sides still predict a show).
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 10
MIN_IMPURITY_DECREASE = 0.01

# Stated volume for the yearly figure: 20 slots a day, 5 days, 50 weeks.
APPOINTMENTS_PER_YEAR = 20 * 5 * 50

EMPTY_SLOT = -120
FALSE_OVERBOOK = -60

D = {
    "lead_time_days": 21,
    "prior_no_shows": 1,
    "age": 27,
    "distance_miles": 9.5,
    "sms_reminder": 0,
    "new_patient": 1,
    "appointment_hour": 15,
}
E = {
    "lead_time_days": 3,
    "prior_no_shows": 0,
    "age": 61,
    "distance_miles": 2.0,
    "sms_reminder": 1,
    "new_patient": 0,
    "appointment_hour": 9,
}


def slot_decision(appointment: dict, model: DecisionTreeClassifier) -> str:
    """Overbook if the tree predicts a no-show. Otherwise hold the slot."""
    row = pd.DataFrame([appointment])[FEATURES]
    predicted_no_show = int(model.predict(row)[0]) == 1
    return "overbook" if predicted_no_show else "hold"


def leaf_for(appointment: dict, model: DecisionTreeClassifier) -> dict:
    tree = model.tree_
    row = pd.Series(appointment)
    node = 0
    while tree.feature[node] != -2:
        feature = FEATURES[tree.feature[node]]
        threshold = float(tree.threshold[node])
        if row[feature] <= threshold:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
    counts = tree.value[node][0]
    no_show_rate = float(counts[1] / counts.sum())
    n = int(tree.n_node_samples[node])
    return {
        "n": n,
        "no_shows": int(round(no_show_rate * n)),
        "no_show_rate": no_show_rate,
        "decision": slot_decision(appointment, model),
    }


def value_of(tn: int, fp: int, fn: int, tp: int) -> float:
    return EMPTY_SLOT * fn + FALSE_OVERBOOK * fp


def pack_fit(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    total = value_of(tn, fp, fn, tp)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "predicted_positive": int(fp + tp),
        "value": float(total),
        "value_per": float(total / len(y_true)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if len(df) != 400:
        raise SystemExit(f"Expected 400 appointments, found {len(df)}")

    y = (df["no_show"] == "no-show").astype(int)
    X = df[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_impurity_decrease=MIN_IMPURITY_DECREASE,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    used = [
        FEATURES[i]
        for i, importance in enumerate(model.feature_importances_)
        if importance > 0
    ]
    train_rows = X_train.copy()
    train_rows["y"] = y_train.to_numpy()
    train_rows["leaf"] = model.apply(X_train)
    leaves = []
    for leaf_id, part in train_rows.groupby("leaf"):
        leaves.append(
            {
                "n": int(len(part)),
                "no_shows": int(part["y"].sum()),
                "decision": slot_decision(part.iloc[0].to_dict(), model),
            }
        )
    train_fit = pack_fit(y_train, model.predict(X_train))
    test_fit = pack_fit(y_test, model.predict(X_test))
    baseline_pred = pd.Series(0, index=y_test.index)
    baseline_fit = pack_fit(y_test, baseline_pred)

    per_gap = test_fit["value_per"] - baseline_fit["value_per"]
    facts = {
        "n": int(len(df)),
        "n_no_show": int(y.sum()),
        "no_show_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_no_show_rate": float(y_train.mean()),
        "test_no_show_rate": float(y_test.mean()),
        "train_no_shows": int(y_train.sum()),
        "test_no_shows": int(y_test.sum()),
        "random_state": RANDOM_STATE,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "features": FEATURES,
        "splits_on": used,
        "leaves": leaves,
        "train_accuracy": train_fit["accuracy"],
        "test": test_fit,
        "baseline": baseline_fit,
        "per_gap": float(per_gap),
        "annual_appointments": APPOINTMENTS_PER_YEAR,
        "annual_value": float(per_gap * APPOINTMENTS_PER_YEAR),
        "D": leaf_for(D, model),
        "E": leaf_for(E, model),
    }
    print("splits", used)
    print("train acc", round(train_fit["accuracy"], 4), "test", round(test_fit["accuracy"], 4))
    print("baseline", baseline_fit)
    print("v1", test_fit)
    print("D", facts["D"])
    print("E", facts["E"])
    print("per gap", per_gap, "year", facts["annual_value"])
    write_slides(facts, HERE / "slides.pdf")
    print("wrote", HERE / "slides.pdf")


if __name__ == "__main__":
    main()
