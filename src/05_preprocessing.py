from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType,
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)
from pyspark.sql.functions import col, when
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)


# --------------------------------------------------
# CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("House Price Prediction - Preprocessing")
    .getOrCreate()
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("nullValue", "NA")
    .csv("data/train.csv")
)


# --------------------------------------------------
# IDENTIFY FEATURE TYPES
# --------------------------------------------------

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

    column_name = field.name

    if column_name == "SalePrice":
        continue

    if isinstance(field.dataType, numerical_types):
        numerical_columns.append(column_name)

    elif isinstance(field.dataType, StringType):
        categorical_columns.append(column_name)


print("\nFEATURE SUMMARY")
print("-" * 50)

print("Numerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))


# --------------------------------------------------
# FILL CATEGORICAL MISSING VALUES
# --------------------------------------------------

df = df.fillna(
    "Unknown",
    subset=categorical_columns
)


# --------------------------------------------------
# NUMERICAL IMPUTATION
# --------------------------------------------------

imputer = Imputer(
    inputCols=numerical_columns,
    outputCols=numerical_columns,
    strategy="median"
)


df = imputer.fit(df).transform(df)


# --------------------------------------------------
# STRING INDEXING
# --------------------------------------------------

indexed_columns = [
    column + "_indexed"
    for column in categorical_columns
]


indexer = StringIndexer(
    inputCols=categorical_columns,
    outputCols=indexed_columns,
    handleInvalid="keep"
)


df = indexer.fit(df).transform(df)


# --------------------------------------------------
# ONE HOT ENCODING
# --------------------------------------------------

encoded_columns = [
    column + "_encoded"
    for column in categorical_columns
]


encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns
)


df = encoder.fit(df).transform(df)


# --------------------------------------------------
# ASSEMBLE FEATURES
# --------------------------------------------------

feature_columns = (
    numerical_columns +
    encoded_columns
)


assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)


final_df = assembler.transform(df)


# --------------------------------------------------
# SELECT FINAL DATASET
# --------------------------------------------------

final_df = final_df.select(
    col("features"),
    col("SalePrice").alias("label")
)


print("\nFINAL DATASET")
print("-" * 50)

print("Rows:", final_df.count())

print("Feature vector size:")

first_row = final_df.first()

print(first_row.features.size)


print("\nSAMPLE ROW")

final_df.show(5, truncate=False)


spark.stop()