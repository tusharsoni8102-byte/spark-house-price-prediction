import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# MODEL RESULTS
# --------------------------------------------------

models = [
    "Linear Regression",
    "Random Forest",
    "Gradient-Boosted Trees"
]

rmse = [
    27188.64,
    28441.83,
    39081.43
]

mae = [
    17934.66,
    17648.43,
    23390.16
]

r2 = [
    0.878,
    0.890,
    0.748
]


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# RMSE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(models, rmse)

plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "outputs/model_rmse_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# MAE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(models, mae)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("MAE")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "outputs/model_mae_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# R2 COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(models, r2)

plt.title("Model Comparison - R² Score")
plt.xlabel("Model")
plt.ylabel("R² Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "outputs/model_r2_comparison.png",
    dpi=300
)

plt.close()


print("\nModel comparison charts created successfully.")
print("Saved in the outputs folder:")
print("- model_rmse_comparison.png")
print("- model_mae_comparison.png")
print("- model_r2_comparison.png")