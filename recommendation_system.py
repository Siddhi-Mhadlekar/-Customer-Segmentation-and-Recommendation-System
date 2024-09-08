from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
# Prepare data for ALS model (user_id, product_id, and reordered)
als_data = cleaned_df.select("user_id", "product_id", "reordered").withColumn("reordered", col("reordered").cast("float"))

# Split the data into training and test sets
(train_data, test_data) = als_data.randomSplit([0.8, 0.2])

# Build the ALS model
als = ALS(userCol="user_id", itemCol="product_id", ratingCol="reordered", coldStartStrategy="drop", nonnegative=True)
als_model = als.fit(train_data)

# Generate product recommendations for all users
user_recommendations = als_model.recommendForAllUsers(5)

# Show a sample of recommendations
user_recommendations.show(5, truncate=False)


# Evaluate the ALS model using Root Mean Squared Error (RMSE)
predictions = als_model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="reordered", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"Root-mean-square error = {rmse}")
