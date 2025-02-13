from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("assignment_yellow_taxi_demand_fare") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

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

# File paths
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

# Doing union
df_union = df_list[0]
for df in df_list[1:]:
    df_union = df_union.unionByName(df, allowMissingColumns=True)

# DataFrame to RDD conversion for MapReduce
rdd = df_union.rdd

def map_metrics(row):
    """Extract key-value pairs from each row for MapReduce."""
    results = []
    if row["tip_amount"] is not None:
        results.append(("avg_tip", (row["tip_amount"], 1)))
    if row["trip_distance"] is not None:
        results.append(("avg_trip_distance", (row["trip_distance"], 1)))
    if row["tolls_amount"] is not None:
        results.append(("total_tolls", (row["tolls_amount"], 1)))
    return results

# Apply the map function
mapped_rdd = rdd.flatMap(map_metrics)

# Reduce function
def reduce_metrics(a, b):
    """Reduce function to sum values and counts."""
    return (a[0] + b[0], a[1] + b[1])


reduced_rdd = mapped_rdd.reduceByKey(reduce_metrics)

def compute_final_values(pair):
    """Compute final average values."""
    key, (total, count) = pair
    return key, (total / count if count > 0 else 0)

# transformation of data
final_rdd = reduced_rdd.map(compute_final_values)

# Convert RDD to DataFrame
metrics_df = final_rdd.toDF(["metric", "value"])

# Write results to GCS
output_path = "gs://assignment_yellow_taxi_demand_fare/map_reduce_output"
metrics_df.coalesce(1).write.mode("overwrite").option("header", "true") \
    .csv(output_path)

# Stop Spark session
spark.stop()