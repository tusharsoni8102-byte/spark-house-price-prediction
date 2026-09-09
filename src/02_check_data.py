from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("House Price Prediction - Data Check")
    .getOrCreate()
)

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)

print("\nRows:", df.count())
print("Columns:", len(df.columns))

print("\nChecking selected columns:")

columns_to_check = [
    "LotFrontage",
    "Alley",
    "PoolQC",
    "Fence",
    "FireplaceQu",
    "GarageType"
]

for column in columns_to_check:
    if column in df.columns:
        print(f"\n--- {column} ---")

        df.groupBy(column).count().orderBy(
            F.desc("count")
        ).show(20, truncate=False)

spark.stop()