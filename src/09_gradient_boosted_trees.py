from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StringType


# --------------------------------------------------
# CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("HousePriceGradientBoostedTrees")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)


# --------------------------------------------------
# IDENTIFY FEATURE TYPES
# --------------------------------------------------

target_column = "SalePrice"

numerical_columns = [
    field.name
    for field in data.schema.fields
    if not isinstance(field.dataType, StringType)
    and field.name != target_column
]

categorical_columns = [
    field.name
    for field in data.schema.fields
    if isinstance(field.dataType, StringType)
]


print("\nFEATURE SUMMARY")
print("-" * 50)
print(f"Numerical features: {len(numerical_columns)}")
print(f"Categorical features: {len(categorical_columns)}")


# --------------------------------------------------
# HANDLE NUMERICAL MISSING VALUES
# --------------------------------------------------

numerical_imputed_columns = [
    f"{column}_imputed"
    for column in numerical_columns
]

numerical_imputer = Imputer(
    inputCols=numerical_columns,
    outputCols=numerical_imputed_columns
).setStrategy("median")


# --------------------------------------------------
# HANDLE CATEGORICAL MISSING VALUES
# --------------------------------------------------

data = data.fillna(
    "Missing",
    subset=categorical_columns
)


# --------------------------------------------------
# INDEX CATEGORICAL FEATURES
# --------------------------------------------------

indexers = [
    StringIndexer(
        inputCol=column,
        outputCol=f"{column}_index",
        handleInvalid="keep"
    )
    for column in categorical_columns
]


# --------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

indexed_columns = [
    f"{column}_index"
    for column in categorical_columns
]

encoded_columns = [
    f"{column}_encoded"
    for column in categorical_columns
]

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns
)


# --------------------------------------------------
# CREATE FEATURE VECTOR
# --------------------------------------------------

feature_columns = (
    numerical_imputed_columns
    + encoded_columns
)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nDATA SPLIT")
print("-" * 50)
print(f"Training rows: {train_data.count()}")
print(f"Testing rows: {test_data.count()}")


# --------------------------------------------------
# CREATE GRADIENT BOOSTED TREE MODEL
# --------------------------------------------------

gbt = GBTRegressor(
    featuresCol="features",
    labelCol=target_column,
    maxIter=100,
    maxDepth=5,
    seed=42
)


# --------------------------------------------------
# CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    stages=[
        numerical_imputer,
        *indexers,
        encoder,
        assembler,
        gbt
    ]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTRAINING GRADIENT-BOOSTED TREE MODEL...")
print("-" * 50)

model = pipeline.fit(train_data)

print("Model training completed.")


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.transform(test_data)


print("\nPREDICTION SAMPLE")
print("-" * 50)

predictions.select(
    target_column,
    "prediction"
).show(10, truncate=False)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

rmse_evaluator = RegressionEvaluator(
    labelCol=target_column,
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol=target_column,
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol=target_column,
    predictionCol="prediction",
    metricName="r2"
)


rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)


print("\nMODEL EVALUATION")
print("-" * 50)

print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")
print(f"R²: {r2:.4f}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

gbt_model = model.stages[-1]

print("\nMODEL INFORMATION")
print("-" * 50)

print(f"Number of trees: {gbt_model.getNumTrees}")
print(f"Maximum depth: {gbt.getMaxDepth()}")


# --------------------------------------------------
# STOP SPARK
# --------------------------------------------------

spark.stop()