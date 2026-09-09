from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# --------------------------------------------------
# CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("HousePriceRandomForest")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)


# --------------------------------------------------
# DROP ID COLUMN
# --------------------------------------------------

df = df.drop("Id")


# --------------------------------------------------
# CORRECT COLUMNS THAT SHOULD BE NUMERICAL
# --------------------------------------------------

numerical_corrections = [
    "LotFrontage",
    "MasVnrArea",
    "GarageYrBlt"
]

for column_name in numerical_corrections:
    df = df.withColumn(
        column_name,
        col(column_name).cast("double")
    )


# --------------------------------------------------
# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

target_column = "SalePrice"

numerical_columns = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString()
    in ("int", "bigint", "double", "float")
    and field.name != target_column
]

categorical_columns = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString() == "string"
]


print("\nFEATURE SUMMARY")
print("-" * 50)

print(f"Numerical features: {len(numerical_columns)}")
print(f"Categorical features: {len(categorical_columns)}")


# --------------------------------------------------
# HANDLE NUMERICAL MISSING VALUES
# --------------------------------------------------

imputer = Imputer(
    inputCols=numerical_columns,
    outputCols=[
        f"{column_name}_imputed"
        for column_name in numerical_columns
    ]
).setStrategy("median")


imputed_numerical_columns = [
    f"{column_name}_imputed"
    for column_name in numerical_columns
]


# --------------------------------------------------
# HANDLE CATEGORICAL MISSING VALUES
# --------------------------------------------------

for column_name in categorical_columns:
    df = df.withColumn(
        column_name,
        when(
            col(column_name).isNull()
            | isnan(col(column_name)),
            "NA"
        ).otherwise(col(column_name))
    )


# --------------------------------------------------
# INDEX CATEGORICAL COLUMNS
# --------------------------------------------------

indexers = [
    StringIndexer(
        inputCol=column_name,
        outputCol=f"{column_name}_index",
        handleInvalid="keep"
    )
    for column_name in categorical_columns
]


indexed_columns = [
    f"{column_name}_index"
    for column_name in categorical_columns
]


# --------------------------------------------------
# ONE-HOT ENCODE CATEGORICAL COLUMNS
# --------------------------------------------------

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=[
        f"{column_name}_encoded"
        for column_name in categorical_columns
    ],
    handleInvalid="keep"
)


encoded_columns = [
    f"{column_name}_encoded"
    for column_name in categorical_columns
]


# --------------------------------------------------
# CREATE FEATURE VECTOR
# --------------------------------------------------

assembler = VectorAssembler(
    inputCols=imputed_numerical_columns + encoded_columns,
    outputCol="features",
    handleInvalid="keep"
)


# --------------------------------------------------
# CREATE RANDOM FOREST MODEL
# --------------------------------------------------

random_forest = RandomForestRegressor(
    featuresCol="features",
    labelCol=target_column,
    predictionCol="prediction",
    numTrees=100,
    maxDepth=10,
    seed=42
)


# --------------------------------------------------
# CREATE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    stages=[
        imputer,
        *indexers,
        encoder,
        assembler,
        random_forest
    ]
)


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)


print("\nDATA SPLIT")
print("-" * 50)

print(f"Training rows: {train_data.count()}")
print(f"Testing rows: {test_data.count()}")


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTRAINING RANDOM FOREST MODEL...")
print("-" * 50)

model = pipeline.fit(train_data)

print("Model training completed.")


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.transform(test_data)


# --------------------------------------------------
# SHOW SAMPLE PREDICTIONS
# --------------------------------------------------

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

random_forest_model = model.stages[-1]

print("\nMODEL INFORMATION")
print("-" * 50)

print(f"Number of trees: {random_forest_model.getNumTrees}")
print(f"Maximum depth: {random_forest_model.getMaxDepth()}")


# --------------------------------------------------
# STOP SPARK
# --------------------------------------------------

spark.stop()