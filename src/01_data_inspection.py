from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .appName("House Price Prediction - Data Inspection")
    .getOrCreate()
)

print("Spark version:", spark.version)

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True
)

print("\nDataset information")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\nSchema:")
df.printSchema()

print("\nFirst 5 rows:")
df.show(5, truncate=False)

print("\nSalePrice statistics:")
df.select("SalePrice").describe().show()

spark.stop()