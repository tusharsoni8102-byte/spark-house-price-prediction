from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)

spark = SparkSession.builder \
    .appName("HousePriceTrainTestSplit") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# -----------------------------
# LOAD DATA
# -----------------------------

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)


# -----------------------------
# CORRECT FEATURE TYPES
# -----------------------------

numeric_columns = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
]

for column in numeric_columns:
    df = df.withColumn(
        column,
        df[column].cast("double")
    )


# -----------------------------
# IDENTIFY FEATURE TYPES
# -----------------------------

target_column = "SalePrice"

numeric_types = (
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)

numerical_columns = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, numeric_types)
    and field.name != target_column
]

categorical_columns = [
    field.name
    for field in df.schema.fields
    if field.name not in numerical_columns
    and field.name != target_column
]


print("\nFEATURE SUMMARY")
print("-" * 50)

print("Numerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))


# -----------------------------
# HANDLE NUMERICAL MISSING VALUES
# -----------------------------

imputer = Imputer(
    inputCols=numerical_columns,
    outputCols=numerical_columns
).setStrategy("median")


# -----------------------------
# HANDLE CATEGORICAL MISSING VALUES
# -----------------------------

for column in categorical_columns:
    df = df.fillna("NA", subset=[column])


# -----------------------------
# STRING INDEXING
# -----------------------------

indexers = []

indexed_columns = []

for column in categorical_columns:

    indexed_column = column + "_index"

    indexer = StringIndexer(
        inputCol=column,
        outputCol=indexed_column,
        handleInvalid="keep"
    )

    indexers.append(indexer)

    indexed_columns.append(indexed_column)


# -----------------------------
# ONE HOT ENCODING
# -----------------------------

encoded_columns = [
    column + "_encoded"
    for column in categorical_columns
]

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns
)


# -----------------------------
# ASSEMBLE FEATURES
# -----------------------------

assembler = VectorAssembler(
    inputCols=numerical_columns + encoded_columns,
    outputCol="features",
    handleInvalid="keep"
)


# -----------------------------
# CREATE PIPELINE
# -----------------------------

pipeline = Pipeline(
    stages=[
        imputer
    ] + indexers + [
        encoder,
        assembler
    ]
)


# -----------------------------
# TRANSFORM DATA
# -----------------------------

model = pipeline.fit(df)

processed_df = model.transform(df)


# -----------------------------
# SELECT FEATURES + LABEL
# -----------------------------

final_df = processed_df.select(
    "features",
    target_column
).withColumnRenamed(
    target_column,
    "label"
)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

train_df, test_df = final_df.randomSplit(
    [0.8, 0.2],
    seed=42
)


print("\nTRAIN / TEST SPLIT")
print("-" * 50)

print("Total rows:", final_df.count())

print("Training rows:", train_df.count())

print("Testing rows:", test_df.count())


print("\nTRAINING DATA SAMPLE")

train_df.show(
    5,
    truncate=False
)


print("\nTEST DATA SAMPLE")

test_df.show(
    5,
    truncate=False
)


spark.stop()