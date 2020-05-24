package me.sparker0i.machinelearning.transformation

import me.sparker0i.machinelearning.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors

object Normalization extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)
    import spark.implicits._

    val points = for (i <- 1 to 1000) yield (i, Vectors.dense(
        Array(
            (math.random() * (10 - 1)) * i + 1.0,
            (math.random() * (10000 - 1000)) + 1000.0,
            math.random() * i
        )
    ))

    val featuresDf = points.toDF("id", "features")

    featuresDf.show(truncate = false)

    val featureScaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("sfeatures")

    val sModel = featureScaler.fit(featuresDf)
    val sFeaturesDf = sModel.transform(featuresDf)

    sFeaturesDf.show(truncate = false)
}
