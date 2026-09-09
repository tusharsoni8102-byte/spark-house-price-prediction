from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import (
    IntegerType,
    DoubleType,
    LongType,
    FloatType,
    ShortType
)


# ---------------------------------
# CREATE SPARK SESSION
# ---------------------------------

spark = SparkSession.builder \
    .appName("HousePriceLinearRegression") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------
# LOAD DATA
# ---------------------------------

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)


# ---------------------------------
# CORRECT FEATURE TYPES
# ---------------------------------

numeric_columns_to_cast = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
]

for column in numeric_columns_to_cast:
    df = df.withColumn(
        column,
        df[column].cast("double")
    )


# ---------------------------------
# IDENTIFY FEATURE TYPES
# ---------------------------------

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


# ---------------------------------
# HANDLE NUMERICAL MISSING VALUES
# ---------------------------------

imputer = Imputer(
    inputCols=numerical_columns,
    outputCols=numerical_columns
).setStrategy("median")


# ---------------------------------
# HANDLE CATEGORICAL MISSING VALUES
# ---------------------------------

for column in categorical_columns:
    df = df.fillna(
        "NA",
        subset=[column]
    )


# ---------------------------------
# STRING INDEXING
# ---------------------------------

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


# ---------------------------------
# ONE-HOT ENCODING
# ---------------------------------

encoded_columns = [
    column + "_encoded"
    for column in categorical_columns
]

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns
)


# ---------------------------------
# FEATURE ASSEMBLER
# ---------------------------------

assembler = VectorAssembler(
    inputCols=numerical_columns + encoded_columns,
    outputCol="features",
    handleInvalid="keep"
)


# ---------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------

preprocessing_pipeline = Pipeline(
    stages=[
        imputer
    ] + indexers + [
        encoder,
        assembler
    ]
)


# ---------------------------------
# SPLIT RAW DATA FIRST
# ---------------------------------

train_raw, test_raw = df.randomSplit(
    [0.8, 0.2],
    seed=42
)


print("\nDATA SPLIT")
print("-" * 50)

print("Training rows:", train_raw.count())
print("Testing rows:", test_raw.count())


# ---------------------------------
# FIT PREPROCESSING ON TRAIN DATA
# ---------------------------------

preprocessing_model = preprocessing_pipeline.fit(
    train_raw
)


# ---------------------------------
# TRANSFORM TRAIN AND TEST DATA
# ---------------------------------

train_processed = preprocessing_model.transform(
    train_raw
)

test_processed = preprocessing_model.transform(
    test_raw
)


# ---------------------------------
# SELECT FINAL COLUMNS
# ---------------------------------

train_df = train_processed.select(
    "features",
    target_column
).withColumnRenamed(
    target_column,
    "label"
)


test_df = test_processed.select(
    "features",
    target_column
).withColumnRenamed(
    target_column,
    "label"
)


# ---------------------------------
# CREATE LINEAR REGRESSION MODEL
# ---------------------------------

lr = LinearRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.0,
    elasticNetParam=0.0
)


# ---------------------------------
# TRAIN MODEL
# ---------------------------------

print("\nTRAINING LINEAR REGRESSION MODEL...")
print("-" * 50)

lr_model = lr.fit(
    train_df
)


print("Model training completed.")


# ---------------------------------
# MAKE PREDICTIONS
# ---------------------------------

predictions = lr_model.transform(
    test_df
)


print("\nPREDICTION SAMPLE")
print("-" * 50)

predictions.select(
    "label",
    "prediction"
).show(
    10,
    truncate=False
)


# ---------------------------------
# EVALUATE MODEL
# ---------------------------------

rmse_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)


rmse = rmse_evaluator.evaluate(
    predictions
)

mae = mae_evaluator.evaluate(
    predictions
)

r2 = r2_evaluator.evaluate(
    predictions
)


# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------

print("\nMODEL EVALUATION")
print("-" * 50)

print(f"RMSE: {rmse:,.2f}")

print(f"MAE: {mae:,.2f}")

print(f"R²: {r2:.4f}")


# ---------------------------------
# MODEL COEFFICIENT INFORMATION
# ---------------------------------

print("\nMODEL INFORMATION")
print("-" * 50)

print("Number of coefficients:",
      len(lr_model.coefficients)
)

print("Intercept:",
      lr_model.intercept
)


spark.stop()