import pyspark.sql
import pytest
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

from features.token_features import compute_features_fn


@pytest.fixture(scope="session")
def spark(request):
    """ fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-pyspark-local-testing") \
        .getOrCreate()
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.mark.usefixtures("spark")
def test_token_features_fn(spark):
    
    input_df = pd.DataFrame(
        {
            "InvoiceNo": ["536365"],
            "StockCode": ["85123A"],
            "Description": ["WHITE HANGING HEART T-LIGHT HOLDER "],
            "Quantity": ["6"],
            "InvoiceDate": ["12/1/10 8:26"],
            "UnitPrice": ["2.55"],
            "CustomerID": ["17850"],
            "Country": ["United Kingdom"]
        }
    )
    spark_df = spark.createDataFrame(input_df)
    output_df = compute_features_fn(spark_df, None, None, None, spark=spark)
    assert isinstance(output_df, pyspark.sql.DataFrame)
    assert output_df.count() == 1 # 4 15-min intervals over 1 hr window.
