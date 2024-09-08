from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CustomerSegmentationBatch") \
    .getOrCreate()

# Load the orders dataset
orders = spark.read.csv("path_to_data/orders.csv", header=True, inferSchema=True)
order_products = spark.read.csv("path_to_data/order_products__prior.csv", header=True, inferSchema=True)
products = spark.read.csv("path_to_data/products.csv", header=True, inferSchema=True)

# Join orders and order_products data
orders_products = orders.join(order_products, "order_id")

# Clean and preprocess data
cleaned_df = orders_products.select("user_id", "product_id", "add_to_cart_order", "reordered")

# Show a sample of the data
cleaned_df.show(5)
