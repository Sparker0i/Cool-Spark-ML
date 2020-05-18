package me.sparker0i.machinelearning.transformation

import me.sparker0i.machinelearning.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors

object Normalization extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val featuresDf = spark.createDataFrame{
        Array(
            (1, Vectors.dense(Array(10.0, 1000.0, 1.0))),
            (2, Vectors.dense(Array(20.0, 3000.0, 2.0))),
            (3, Vectors.dense(Array(30.0, 4000.0, 3.0)))
        )
    }
        .withColumnRenamed("_1", "id")
        .withColumnRenamed("_2", "features")

    featuresDf.show()

    val featureScaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("sfeatures")

    val sModel = featureScaler.fit(featuresDf)
    val sFeaturesDf = sModel.transform(featuresDf)

    sFeaturesDf.show()
}
