from pyspark.sql.functions import countDistinct, sum, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Feature engineering: Calculate features like total orders, distinct products, and reorder ratio
customer_features = cleaned_df.groupBy("user_id") \
    .agg(countDistinct("order_id").alias("total_orders"),
         countDistinct("product_id").alias("distinct_products"),
         avg("reordered").alias("reorder_ratio"))

# Show the features for customers
customer_features.show(5)




# Assemble features into a feature vector
assembler = VectorAssembler(inputCols=["total_orders", "distinct_products", "reorder_ratio"], outputCol="features")
customer_vector = assembler.transform(customer_features)

# Apply K-Means clustering
kmeans = KMeans(k=3, seed=1)  # Set number of clusters
model = kmeans.fit(customer_vector)

# Make predictions and assign cluster labels
clustered_customers = model.transform(customer_vector)

# Show customers with their assigned cluster
clustered_customers.select("user_id", "prediction").show(5)
