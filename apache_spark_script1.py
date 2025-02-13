from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# Initializing Spark session
spark = SparkSession.builder.appName("assignment_yellow_taxi_demand_fare").getOrCreate()

# Define schema for performance optimization
schema = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", StringType(), True),
    StructField("tpep_dropoff_datetime", StringType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", FloatType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("PULocationID", IntegerType(), True),
    StructField("DOLocationID", IntegerType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", FloatType(), True),
    StructField("extra", FloatType(), True),
    StructField("mta_tax", FloatType(), True),
    StructField("tip_amount", FloatType(), True),
    StructField("tolls_amount", FloatType(), True),
    StructField("improvement_surcharge", FloatType(), True),
    StructField("total_amount", FloatType(), True),
    StructField("congestion_surcharge", FloatType(), True),
    StructField("Airport_fee", FloatType(), True)
])

# All the F=file paths
file_paths = [
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-09.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-10.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2023-11.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-09.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-10.csv",
    "gs://assignment_yellow_taxi_demand_fare/yellow_tripdata_2024-11.csv"
]

# Read and preprocess data
df_list = [spark.read.option("header", "true").schema(schema).csv(file) for file in file_paths]

# Union all DataFrames using unionByName to handle column mismatches
df_union = df_list[0]
for df in df_list[1:]:
    df_union = df_union.unionByName(df, allowMissingColumns=True)

# Print total row count
total_rows = df_union.count()
print(f"Total Rows: {total_rows}")

# Remove duplicates
df_union = df_union.dropDuplicates()

# Drop rows with missing critical values
df_union = df_union.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount"])

# Recalculate the total rows after cleaning
cleaned_total_rows = df_union.count()
print(f"Total Rows After Cleaning: {cleaned_total_rows}")

# Compute multiple statistics
metrics = {
    "avg_tip": df_union.select(F.avg("tip_amount").alias("avg_tip")),
    "avg_trip_distance": df_union.select(F.avg("trip_distance").alias("avg_trip_distance")),
    "total_tolls": df_union.select(F.sum("tolls_amount").alias("total_tolls"))
}

# Write results to GCS using a loop
output_path = "gs://assignment_yellow_taxi_demand_fare/output_new"
for metric_name, df in metrics.items():
    df.coalesce(1).write.mode("overwrite").option("header", "true") \
        .csv(f"{output_path}/{metric_name}")
    

# Stop Spark session
spark.stop()