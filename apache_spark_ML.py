from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid")

# Initialize Spark
spark = SparkSession.builder \
    .appName("TaxiFarePrediction_GBT") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.memoryOverhead", "512m") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# Fiel paths
file_paths = [
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-09.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-10.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-11.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-09.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-10.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-11.csv"
]

# Load data
df_base = spark.read.csv(file_paths[0], header=True, inferSchema=True)
standard_columns = df_base.columns  

dfs = []
for path in file_paths:
    df_temp = spark.read.csv(path, header=True, inferSchema=True)
    for col in standard_columns:
        if col not in df_temp.columns:
            df_temp = df_temp.withColumn(col, F.lit(None).cast(df_base.schema[col].dataType))
    df_temp = df_temp.select(standard_columns)
    dfs.append(df_temp)

df = dfs[0]
for df_temp in dfs[1:]:
    df = df.union(df_temp)

# drop dupicated
df = df.dropDuplicates().repartition(20)

sample_df = df.sample(False, 0.01, seed=42).toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(sample_df["total_amount"], bins=50, kde=True, color='blue')
plt.title("Distribution of Total Fare Amounts")
plt.xlabel("Total Amount ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("fare_amount_distribution.png")

plt.figure(figsize=(10, 6))
sns.scatterplot(x="trip_distance", y="total_amount", data=sample_df, alpha=0.5)
plt.title("Trip Distance vs. Total Fare Amount")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Total Amount ($)")
plt.grid(True)
plt.savefig("trip_distance_vs_fare.png")

df = df.filter((df["total_amount"] > 2) & (df["total_amount"] < 200))

indexers = [
    StringIndexer(inputCol="payment_type", outputCol="payment_type_index", handleInvalid="skip"),
    StringIndexer(inputCol="VendorID", outputCol="VendorID_index", handleInvalid="skip"),
    StringIndexer(inputCol="RatecodeID", outputCol="RatecodeID_index", handleInvalid="skip")
]

# Feature Engineering
feature_cols = [
    "passenger_count", "trip_distance", "RatecodeID_index", "PULocationID", "DOLocationID",
    "payment_type_index", "VendorID_index", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "congestion_surcharge"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ML Algo GBT Regressor (Gradient-Boosted Trees)
gbt = GBTRegressor(featuresCol="features", labelCol="total_amount", maxIter=30, maxDepth=5)

# ML Pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt])

# Train/Test Split on 80-20%
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the Model
model = pipeline.fit(train_data)

predictions = model.transform(test_data).select("total_amount", "prediction")

preds_pd = predictions.limit(500).toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x="total_amount", y="prediction", data=preds_pd, color='purple', alpha=0.6)
plt.plot([0, 200], [0, 200], color='red', linestyle='--')  
plt.title("Actual vs. Predicted Total Fare Amounts")
plt.xlabel("Actual Total Amount ($)")
plt.ylabel("Predicted Total Amount ($)")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")

preds_pd["error"] = preds_pd["total_amount"] - preds_pd["prediction"]
plt.figure(figsize=(10, 6))
sns.histplot(preds_pd["error"], bins=50, kde=True, color='orange')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("prediction_errors.png")

# Model Evaluation
rmse_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
r2_evaluator = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")

rmse = rmse_evaluator.evaluate(predictions)
r2 = r2_evaluator.evaluate(predictions)

print(f"ðŸ“‰ Root Mean Squared Error (RMSE): {rmse}")
print(f"ðŸ“ˆ R-squared (R2): {r2}")

spark.stop()