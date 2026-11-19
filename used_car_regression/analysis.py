"""Used-car sale price regression.

One command reproduces the coefficients, the test-set fit, the auction
decision, and the slides:

    python3 analysis.py

Split: 70% train / 30% test, random_state=42.
The regression is fit on the training cars only.
The 40 auction lots are not used to fit anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from build_slides import write_slides

RANDOM_STATE = 42
PREP_COST = 1500
HERE = Path(__file__).resolve().parent
SALES_PATH = HERE / "used_car_sales.csv"
LOTS_PATH = HERE / "auction_lots.csv"

# Num_Owners is fit once, then dropped: its p-value is about 0.87 and
# adjusted R² does not fall without it. Mileage and age are correlated,
# but dropping either one makes the test error worse, so both stay.
FULL_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
V1_FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident"]


def design(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    columns = {}
    for feature in features:
        if feature == "Accident":
            columns["Accident_yes"] = (df["Accident"] == "yes").astype(float)
        else:
            columns[feature] = df[feature].astype(float)
    return sm.add_constant(pd.DataFrame(columns, index=df.index))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def auction_result(predicted_sale: np.ndarray, lots: pd.DataFrame) -> dict:
    """Buy when the predicted lot price clears the auction price plus prep."""
    buy = predicted_sale > lots["Auction_Price"].to_numpy() + PREP_COST
    profit = (
        lots["Later_Sale_Price"].to_numpy()
        - lots["Auction_Price"].to_numpy()
        - PREP_COST
    )
    bought = profit[buy]
    return {
        "bought": int(buy.sum()),
        "losers": int((bought < 0).sum()),
        "profit": float(bought.sum()) if len(bought) else 0.0,
    }


def coef_table(model) -> list[dict]:
    rows = []
    for name in model.params.index:
        rows.append(
            {
                "name": name,
                "coef": float(model.params[name]),
                "pvalue": float(model.pvalues[name]),
            }
        )
    return rows


def main() -> None:
    sales = pd.read_csv(SALES_PATH)
    lots = pd.read_csv(LOTS_PATH)
    if len(sales) != 240:
        raise SystemExit(f"Expected 240 sold cars, found {len(sales)}")
    if len(lots) != 40:
        raise SystemExit(f"Expected 40 auction lots, found {len(lots)}")

    train, test = train_test_split(
        sales, test_size=0.30, random_state=RANDOM_STATE
    )
    y_train = train["Sale_Price"].astype(float)
    y_test = test["Sale_Price"].astype(float)
    train_mean = float(y_train.mean())

    full = sm.OLS(y_train, design(train, FULL_FEATURES)).fit()
    v1 = sm.OLS(y_train, design(train, V1_FEATURES)).fit()

    pred_v1 = v1.predict(design(test, V1_FEATURES)).to_numpy()
    pred_base = np.full(len(test), train_mean)

    baseline_auction = auction_result(np.full(len(lots), train_mean), lots)
    v1_auction = auction_result(
        v1.predict(design(lots, V1_FEATURES)).to_numpy(), lots
    )

    facts = {
        "n_sales": int(len(sales)),
        "n_lots": int(len(lots)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "random_state": RANDOM_STATE,
        "prep_cost": PREP_COST,
        "train_mean": train_mean,
        "full_adj_r2": float(full.rsquared_adj),
        "v1_adj_r2": float(v1.rsquared_adj),
        "v1_r2_train": float(v1.rsquared),
        "owners_pvalue": float(full.pvalues["Num_Owners"]),
        "coefs": coef_table(v1),
        "test_r2_base": float(r2_score(y_test, pred_base)),
        "test_rmse_base": rmse(y_test, pred_base),
        "test_r2_v1": float(r2_score(y_test, pred_v1)),
        "test_rmse_v1": rmse(y_test, pred_v1),
        "baseline_auction": baseline_auction,
        "v1_auction": v1_auction,
        "actual": y_test.to_numpy(),
        "predicted": pred_v1,
    }

    print(full.summary())
    print(v1.summary())
    print(
        f"test baseline R2 {facts['test_r2_base']:.4f} "
        f"RMSE {facts['test_rmse_base']:.1f}"
    )
    print(
        f"test v1 R2 {facts['test_r2_v1']:.4f} "
        f"RMSE {facts['test_rmse_v1']:.1f}"
    )
    print("baseline auction", baseline_auction)
    print("v1 auction", v1_auction)
    write_slides(facts, HERE / "slides.pdf")
    print("wrote", HERE / "slides.pdf")


if __name__ == "__main__":
    main()
