from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType,
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)


spark = (
    SparkSession.builder
    .appName("House Price Prediction - Correct Schema")
    .getOrCreate()
)


df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("nullValue", "NA")
    .csv("data/train.csv")
)


numerical_columns = []
categorical_columns = []

numerical_types = (
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)


for field in df.schema.fields:

    column = field.name

    if column == "SalePrice":
        continue

    if isinstance(field.dataType, numerical_types):
        numerical_columns.append(column)

    elif isinstance(field.dataType, StringType):
        categorical_columns.append(column)


print("\nNUMERICAL COLUMNS")
print("-" * 50)

for column in numerical_columns:
    print(column)


print("\nCATEGORICAL COLUMNS")
print("-" * 50)

for column in categorical_columns:
    print(column)


print("\nSUMMARY")
print("-" * 50)

print("Numerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))
print("Target column: SalePrice")


print("\nCHECKING CORRECTED COLUMNS")

df.select(
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
).printSchema()


print("\nMissing value counts for selected columns:")

for column in ["LotFrontage", "MasVnrArea", "GarageYrBlt"]:
    print(
        column,
        ":",
        df.filter(df[column].isNull()).count()
    )


spark.stop()