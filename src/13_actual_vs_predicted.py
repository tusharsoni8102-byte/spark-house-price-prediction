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

import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("ActualVsPredictedHousePrices")
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

target_column = "SalePrice"


# --------------------------------------------------
# IDENTIFY FEATURE TYPES
# --------------------------------------------------

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


# --------------------------------------------------
# NUMERICAL MISSING VALUES
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
# CATEGORICAL MISSING VALUES
# --------------------------------------------------

data = data.fillna(
    "Missing",
    subset=categorical_columns
)


# --------------------------------------------------
# STRING INDEXING
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
# ONE-HOT ENCODING
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
# FEATURE ASSEMBLY
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
# TRAIN / TEST SPLIT
# --------------------------------------------------

train_data, test_data = data.randomSplit(
    [0.8, 0.2],
    seed=42
)


# --------------------------------------------------
# RANDOM FOREST MODEL
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

print("\nTRAINING RANDOM FOREST MODEL...")

model = pipeline.fit(train_data)


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

predictions = model.transform(test_data)

prediction_data = predictions.select(
    target_column,
    "prediction"
).toPandas()


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# ACTUAL VS PREDICTED SCATTER PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    prediction_data[target_column],
    prediction_data["prediction"],
    alpha=0.6
)

minimum = min(
    prediction_data[target_column].min(),
    prediction_data["prediction"].min()
)

maximum = max(
    prediction_data[target_column].max(),
    prediction_data["prediction"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.xlabel("Actual Sale Price")

plt.ylabel("Predicted Sale Price")

plt.title(
    "Actual vs Predicted House Prices - Random Forest"
)

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300
)

plt.close()


print("\nChart created successfully.")
print("Saved as: outputs/actual_vs_predicted.png")


# --------------------------------------------------
# STOP SPARK
# --------------------------------------------------

spark.stop()