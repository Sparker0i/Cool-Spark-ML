package me.sparker0i.machinelearning.transformation.numeric

import me.sparker0i.machinelearning.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors

object Standardization extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)
    import spark.implicits._

    val points = for (i <- 1 to 1000) yield (i, Vectors.dense(
        Array(
            (math.random * (10 - 1)) * i + 1.0,
            (math.random * (10000 - 1000)) + 1000.0,
            math.random * i
        )
    ))

    val featuresDf = points.toDF("id", "features")

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
