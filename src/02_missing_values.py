from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


spark = (
    SparkSession.builder
    .appName("House Price Prediction - Missing Values")
    .getOrCreate()
)

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)

missing_values = []

for field in df.schema.fields:

    column = field.name

    if isinstance(field.dataType, StringType):

        missing_count = df.filter(
            F.col(column).isNull()
            | (F.trim(F.col(column)) == "")
            | (F.col(column) == "NA")
        ).count()

    else:

        missing_count = df.filter(
            F.col(column).isNull()
        ).count()

    if missing_count > 0:
        missing_values.append(
            (column, missing_count)
        )


print("\nColumns containing missing values:\n")

for column, count in sorted(
    missing_values,
    key=lambda x: x[1],
    reverse=True
):
    print(f"{column}: {count}")


print(f"\nTotal columns with missing values: {len(missing_values)}")

spark.stop()