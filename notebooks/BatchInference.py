# Databricks notebook source
##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# writing output to a delta table. It's scheduled as a batch inference job defined under ``databricks-config``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# A Hive-registered Delta table containing the input features.
dbutils.widgets.text("input_table_name", "invoices", label="Input Table Name")
# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "similar_products_predict", label="Output Table Name")

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys

sys.path.append("../steps")

# COMMAND ----------

from utils import get_deployed_model_stage_for_env, get_model_name
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"

model_name = get_model_name(env)
stage = get_deployed_model_stage_for_env(env)
model_uri = f"models:/{model_name}/{stage}"

# COMMAND ----------

# DBTITLE 1,Define input and output variables
from utils import get_deployed_model_stage_for_env, get_model_name

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"

model_name = get_model_name(env)
stage = get_deployed_model_stage_for_env(env)

# Get model version from stage
from mlflow import MlflowClient

model_version_infos = MlflowClient().search_model_versions("name = '%s'" % model_name)

model_version = max(
    int(version.version) for version in model_version_infos if version.current_stage == stage
)

model_uri = f"models:/{model_name}/{model_version}"

print(model_uri)

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------

# DBTITLE 1,Load model and run inference
from predict import predict_batch
from pyspark.sql.types import DoubleType, ArrayType, StructType, StringType

fs = feature_store.FeatureStoreClient()
raw_data = spark.table(input_table_name)
raw_data=raw_data.na.drop()
products=raw_data[["StockCode"]].drop_duplicates(['StockCode'])

predictions_df = fs.score_batch(
    model_uri, 
    products, 
    ArrayType(DoubleType())
)

predictions_df.display()

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, to_timestamp, lit

output_df = (
    predictions_df
    .withColumn("model_version", lit(model_version))
    .withColumn("inference_timestamp", to_timestamp(lit(ts)))
)

# Model predictions are written to the Delta table provided as input.
# Delta is the default format in Databricks Runtime 8.0 and above.
#output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from product_embeddings_predict limit 5;

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col, udf

def cosine_sim(vecA, vecB):
    """Find the cosine similarity distance between two vectors."""
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    if np.isnan(np.sum(csim)):
        return 0
    return csim

def calculate_similarity(spark_df, threshold=0, limit=5):
    """Calculates & returns similarity scores between given source document & all
    the target documents."""
    df = spark_df.toPandas()[:5]
    df["similar_products"] = None
    for source_index, source_row in df.iterrows():
        results = []
        source_vec=source_row["prediction"]
        for index, row in df.iterrows():
            target_vec = row['prediction']
            sim_score = cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({"score": sim_score, "doc": row["StockCode"]})
        # Sort results by score in desc order
        results.sort(key=lambda k: k["score"], reverse=True)
        results = results[1:limit+1] 
        print(source_index)
        df.at[source_index,'similar_products']=results
        
    return df

def myComplexFunc(row):
    row["description_preprocessed"]=list(row["description_preprocessed"])
    return row

final_df=calculate_similarity(output_df)
final_df.apply(myComplexFunc)
final_df

# COMMAND ----------

from pyspark.sql.types import StructField, DoubleType, ArrayType, StructType, StringType, IntegerType, MapType

schema = StructType([ \
    StructField("StockCode",StringType(),True), \
    StructField("description_preprocessed", ArrayType(StringType()),True), \
    StructField("prediction",ArrayType(DoubleType()),True), \
    StructField("model_version", IntegerType(), True), \
    StructField("inference_timestamp", StringType(), True), \
    StructField("similar_products", ArrayType(MapType(StringType(),StringType(),False)), True) \
  ])

final_spark_df=spark.createDataFrame(final_df, schema=schema) 

# Model predictions are written to the Delta table provided as input.
# Delta is the default format in Databricks Runtime 8.0 and above.
final_spark_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from product_similar_predict;
