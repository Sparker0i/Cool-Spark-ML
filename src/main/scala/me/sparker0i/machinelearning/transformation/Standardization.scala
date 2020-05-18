package me.sparker0i.machinelearning.transformation

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors
import me.sparker0i.machinelearning._
import org.apache.log4j.{Level, Logger}

object Standardization extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val featuresDf = spark.createDataFrame{
        Array(
            (1, Vectors.dense(Array(10.0, 10000.0, 1.0))),
            (2, Vectors.dense(Array(20.0, 30000.0, 2.0))),
            (3, Vectors.dense(Array(30.0, 40000.0, 3.0)))
        )
    }
        .withColumnRenamed("_1", "id")
        .withColumnRenamed("_2", "features")

    featuresDf.show()

    val featureStandardScaler = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("sfeatures")
        .setWithStd(true)
        .setWithMean(true)

    val standSModel = featureStandardScaler.fit(featuresDf)
    val standSFeatures = standSModel.transform(featuresDf)

    standSFeatures.show()
}
