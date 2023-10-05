from pyspark.sql import SparkSession
from pyspark import SparkContext

def get_spark_session()
    spark = (
        pyspark.sql.SparkSession.builder
        .config('spark.executor.memory', '3g')
        .config('spark.executor.cores', '2')
        .config('spark.driver.memory','5g')
        .config('spark.cores.max', '300')
        .getOrCreate()
    )
    return spark