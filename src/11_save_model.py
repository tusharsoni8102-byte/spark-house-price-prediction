from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.types import StringType


# --------------------------------------------------
# CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SaveHousePriceRandomForestModel")
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


# --------------------------------------------------
# CREATE RANDOM FOREST MODEL
# --------------------------------------------------

random_forest = RandomForestRegressor(
    featuresCol="features",
    labelCol=target_column,
    numTrees=100,
    maxDepth=10,
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
        random_forest
    ]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTRAINING FINAL RANDOM FOREST MODEL...")
print("-" * 50)

model = pipeline.fit(train_data)

print("Model training completed.")


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = "outputs/random_forest_house_price_model"

print("\nSAVING MODEL...")
print("-" * 50)

model.write().overwrite().save(model_path)

print(f"Model saved successfully to: {model_path}")


# --------------------------------------------------
# STOP SPARK
# --------------------------------------------------

spark.stop()