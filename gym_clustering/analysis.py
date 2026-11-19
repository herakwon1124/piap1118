"""Gym member k-means segmentation.

One command reproduces the elbow chart, the segment table, the credit
decision, and the slides:

    python3 analysis.py

Split: 70% train / 30% test, random_state=42.
The scaler and k-means are fit on the training members only.
Test members are scored against those training centers and are not used to fit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from build_slides import write_slides

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "gym_members.csv"
K_RANGE = range(1, 9)

# member_id identifies a row. It does not describe the member.
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]

# Elbow: the spread keeps falling through k=4, then flattens.
# k=5 cuts training spread by ~26 and test spread by ~9.
CHOSEN_K = 4

# Names are a reading of the centroids, not a model output.
SEGMENT_NAMES = {
    0: "Newcomers",
    1: "Veterans",
    2: "Floor regulars",
    3: "Studio members",
}

NEW_MEMBER = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}


def credit_net(spend: float, receives_credit: bool) -> float:
    """Finance rules for one member for one month.

    Spend of $20 or more: the credit is a $10 giveaway.
    Spend under $20: extras rise by $25, and 60% margin on that is $15,
    so the $10 credit nets +$5.
    No credit: $0.
    """
    if not receives_credit:
        return 0.0
    if spend >= 20:
        return -10.0
    return 5.0


def wcss(X: np.ndarray, centers: np.ndarray) -> float:
    """Sum of squared distances from each row to its nearest center."""
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).sum())


def fit_kmeans(X_train: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    model.fit(X_train)
    return model


def offer_for_member(features: dict, scaler, model, credit_segments: set) -> dict:
    """Segment and credit decision for one member.

    The segment is the nearest center fit on the training members.
    The credit follows the segment, so a member can be targeted after
    month one without waiting to see a long spend history.
    """
    row = pd.DataFrame([features])[FEATURES]
    segment = int(model.predict(scaler.transform(row))[0])
    name = SEGMENT_NAMES[segment]
    return {
        "segment": segment,
        "segment_name": name,
        "gets_credit": segment in credit_segments,
    }


def trait_text(name: str, means: pd.Series) -> str:
    age = means["age"]
    visits = means["visits_per_month"]
    minutes = means["avg_visit_minutes"]
    classes = means["classes_per_month"]
    months = means["months_member"]
    if name == "Newcomers":
        return (
            f"Age {age:.0f}, joined {months:.0f} months ago, "
            f"{visits:.0f} visits a month"
        )
    if name == "Veterans":
        return (
            f"Age {age:.0f}, member for {months / 12:.0f} years, "
            f"{visits:.0f} visits a month"
        )
    if name == "Floor regulars":
        return (
            f"{visits:.0f} visits of {minutes:.0f} minutes, "
            "under 1 class a month"
        )
    if name == "Studio members":
        return (
            f"Age {age:.0f}, {classes:.0f} classes a month, "
            f"{visits:.0f} visits"
        )
    raise KeyError(name)


def plot_elbow(spread: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(
        spread["k"],
        spread["train_wcss"],
        marker="o",
        color="black",
        label="Training (231 members)",
    )
    ax.plot(
        spread["k"],
        spread["test_wcss"],
        marker="o",
        color="0.45",
        linestyle="--",
        label="Test (99 members, not used to fit)",
    )
    ax.axvline(CHOSEN_K, color="0.75", linewidth=1)
    ax.set_xlabel("Number of groups (k)")
    ax.set_ylabel("Within-group spread")
    ax.set_xticks(list(K_RANGE))
    ax.legend(frameon=False)
    ax.set_title("Sum of squared distances to the group center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if len(df) != 330 or df.shape[1] != 7:
        raise SystemExit(f"Unexpected data shape: {df.shape}")

    train, test = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])

    rows = []
    models = {}
    for k in K_RANGE:
        model = fit_kmeans(X_train, k)
        train_spread = wcss(X_train, model.cluster_centers_)
        if abs(train_spread - model.inertia_) > 1e-6:
            raise SystemExit("Training WCSS does not match KMeans.inertia_")
        models[k] = model
        rows.append(
            {
                "k": k,
                "train_wcss": train_spread,
                "test_wcss": wcss(X_test, model.cluster_centers_),
            }
        )
    spread = pd.DataFrame(rows)
    spread["train_drop"] = spread["train_wcss"].shift(1) - spread["train_wcss"]
    spread["test_drop"] = spread["test_wcss"].shift(1) - spread["test_wcss"]

    model = models[CHOSEN_K]
    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FEATURES,
    )
    centers.index.name = "segment"

    assigned = df.copy()
    assigned["segment"] = model.predict(scaler.transform(df[FEATURES]))
    assigned["segment_name"] = assigned["segment"].map(SEGMENT_NAMES)
    assigned["split"] = np.where(
        assigned["member_id"].isin(set(train["member_id"])), "train", "test"
    )

    value_rows = []
    for seg, part in assigned.groupby("segment"):
        nets = [credit_net(s, True) for s in part["monthly_extras_spend"]]
        value_rows.append(
            {
                "segment": int(seg),
                "segment_name": SEGMENT_NAMES[int(seg)],
                "n": len(part),
                "net_if_credited": float(sum(nets)),
                "n_under_20": int((part["monthly_extras_spend"] < 20).sum()),
                "n_at_least_20": int((part["monthly_extras_spend"] >= 20).sum()),
            }
        )
    value = pd.DataFrame(value_rows)
    credit_segments = set(
        value.loc[value["net_if_credited"] > 0, "segment"].astype(int)
    )

    assigned["gets_credit"] = assigned["segment"].isin(credit_segments)
    assigned["credit_net"] = [
        credit_net(s, flag)
        for s, flag in zip(assigned["monthly_extras_spend"], assigned["gets_credit"])
    ]
    baseline_net = float(
        sum(credit_net(s, True) for s in assigned["monthly_extras_spend"])
    )
    v1_net = float(assigned["credit_net"].sum())

    decision = offer_for_member(NEW_MEMBER, scaler, model, credit_segments)

    # Slide traits use the training centers. Counts and credit dollars use
    # every member, training and test, scored one by one.
    center_means = centers.copy()
    center_means["segment_name"] = center_means.index.map(SEGMENT_NAMES)
    center_means = center_means.set_index("segment_name")
    table_rows = []
    for _, row in value.iterrows():
        name = row["segment_name"]
        table_rows.append(
            {
                "segment_name": name,
                "n": int(row["n"]),
                "traits": trait_text(name, center_means.loc[name]),
                "extras": float(center_means.loc[name, "monthly_extras_spend"]),
                "gets_credit": int(row["segment"]) in credit_segments,
                "net_if_credited": float(row["net_if_credited"]),
            }
        )

    facts = {
        "n_members": int(len(df)),
        "n_columns": int(df.shape[1]),
        "mean_extras": float(df["monthly_extras_spend"].mean()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "left_out": ["member_id"],
        "chosen_k": CHOSEN_K,
        "train_wcss_k1": float(spread.loc[spread.k == 1, "train_wcss"].iloc[0]),
        "train_wcss_k": float(spread.loc[spread.k == CHOSEN_K, "train_wcss"].iloc[0]),
        "test_wcss_k1": float(spread.loc[spread.k == 1, "test_wcss"].iloc[0]),
        "test_wcss_k": float(spread.loc[spread.k == CHOSEN_K, "test_wcss"].iloc[0]),
        "train_drop_4_to_5": float(spread.loc[spread.k == 5, "train_drop"].iloc[0]),
        "test_drop_4_to_5": float(spread.loc[spread.k == 5, "test_drop"].iloc[0]),
        "test_drop_3_to_4": float(spread.loc[spread.k == 4, "test_drop"].iloc[0]),
        "baseline_net": baseline_net,
        "v1_net": v1_net,
        "new_member": decision,
        "segments": table_rows,
    }

    plot_elbow(spread, HERE / "elbow.png")
    spread.to_csv(HERE / "wcss_by_k.csv", index=False)
    centers.round(4).to_csv(HERE / "centroids.csv")
    assigned.to_csv(HERE / "member_segments.csv", index=False)
    value.to_csv(HERE / "segment_value.csv", index=False)
    pd.Series(
        {
            key: facts[key]
            for key in facts
            if key not in {"features", "left_out", "new_member", "segments"}
        }
    ).to_json(HERE / "summary.json")

    write_slides(facts, spread, HERE / "slides.pdf", HERE / "elbow.png")

    print(spread.round(2).to_string(index=False))
    print(centers.round(1).to_string())
    print(value.to_string(index=False))
    print(
        f"baseline ${baseline_net:.0f}  v1 ${v1_net:.0f}  "
        f"lift ${v1_net - baseline_net:.0f}"
    )
    print("new member", decision)
    print("wrote", HERE / "slides.pdf")


if __name__ == "__main__":
    main()
