package me.sparker0i

import org.apache.spark.sql.{DataFrame, SparkSession}

package object machinelearning {
    val spark: SparkSession = SparkSession.builder()
        .appName("Spark-ML")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()

    implicit class DataFramePlus(df: DataFrame) {
        def shape(): (Long, Int) = (df.count(), df.columns.length)
    }
}
