package me.sparker0i.machinelearning.transformation.numeric

import me.sparker0i.machinelearning.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.functions.count

object Bucketize extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val splits = Array(Float.NegativeInfinity, -500.0, -100.0, -10.0, 0.0, 10.0, 100.0, 500.0, Float.PositiveInfinity)

    import spark.implicits._

    val bucketData = (for (i <- 0 to 10000) yield math.random * 10000.0 * (if (math.random < 0.5) -1 else 1))
    val bucketDf = bucketData.toDF("features")

    bucketDf.show()

    val bucketizer = new Bucketizer()
        .setSplits(splits)
        .setInputCol("features")
        .setOutputCol("bfeatures")

    val bucketedDf = bucketizer.transform(bucketDf)
    bucketedDf.show()

    bucketedDf.groupBy("bFeatures").agg(count("*").as("cnt")).orderBy("bFeatures").show()
}
