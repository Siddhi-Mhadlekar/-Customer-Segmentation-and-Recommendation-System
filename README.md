# Customer Segmentation and Recommendation System (Batch Processing)

## Overview
This project demonstrates customer segmentation using clustering (K-Means) and product recommendation using collaborative filtering (ALS) with PySpark in a batch processing setup.

### Steps:
1. **Data Ingestion**: Load and preprocess customer transaction data.
2. **Customer Segmentation**: Apply K-Means clustering to group customers based on purchasing behavior.
3. **Recommendation System**: Use ALS to recommend products to customers based on previous purchases.
4. **Evaluation**: Evaluate the recommendation model using RMSE.
5. **Visualization**: Export results to CSV for visualization and reporting.

### How to Run:
1. Install dependencies:
    ```bash
    pip install pyspark
    ```

2. Run the data ingestion, segmentation, and recommendation scripts:
    ```bash
    python data_ingestion.py
    python customer_segmentation.py
    python recommendation_system.py
    ```

3. Output CSV files will be saved in the `outputs` directory for further analysis.

### Dataset
We use the **Instacart Market Basket Analysis Dataset** for customer transactions.
