# --------------------------------------------------
# HOUSE PRICE MODEL COMPARISON
# --------------------------------------------------

import pandas as pd


# --------------------------------------------------
# MODEL RESULTS
# --------------------------------------------------

results = {
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient-Boosted Trees"
    ],

    "RMSE": [
        27188.64,
        28441.83,
        39081.43
    ],

    "MAE": [
        17934.66,
        17648.43,
        23390.16
    ],

    "R2": [
        0.8780,
        0.8900,
        0.7480
    ]
}


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

comparison = pd.DataFrame(results)


print("\nMODEL COMPARISON")
print("-" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# --------------------------------------------------
# FIND BEST MODELS
# --------------------------------------------------

best_rmse = comparison.loc[
    comparison["RMSE"].idxmin()
]

best_mae = comparison.loc[
    comparison["MAE"].idxmin()
]

best_r2 = comparison.loc[
    comparison["R2"].idxmax()
]


print("\nBEST MODEL BY RMSE")
print("-" * 70)

print(f"Model: {best_rmse['Model']}")
print(f"RMSE: {best_rmse['RMSE']:,.2f}")


print("\nBEST MODEL BY MAE")
print("-" * 70)

print(f"Model: {best_mae['Model']}")
print(f"MAE: {best_mae['MAE']:,.2f}")


print("\nBEST MODEL BY R²")
print("-" * 70)

print(f"Model: {best_r2['Model']}")
print(f"R²: {best_r2['R2']:.4f}")


# --------------------------------------------------
# FINAL MODEL SELECTION
# --------------------------------------------------

print("\nFINAL MODEL SELECTION")
print("-" * 70)

print("Recommended Model: Random Forest Regressor")

print(
    "\nReason: Random Forest achieved the highest R² score "
    "and the lowest MAE among the evaluated models."
)