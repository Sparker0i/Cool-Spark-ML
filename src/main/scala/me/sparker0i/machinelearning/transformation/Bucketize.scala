package me.sparker0i.machinelearning.transformation

import me.sparker0i.machinelearning._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.ml.linalg.Vectors

object Bucketize extends App {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val splits = Array(Float.NegativeInfinity, -10.0, 0.0, 10.0, Float.PositiveInfinity)

    val bucketData = Array(Tuple1(-800.0), Tuple1(-10.5), Tuple1(-1.7), Tuple1(0.0), Tuple1(8.2), Tuple1(90.1))
    val bucketDf = spark.createDataFrame(bucketData)
        .withColumnRenamed("_1", "features")

    bucketDf.show()

    val bucketizer = new Bucketizer()
        .setSplits(splits)
        .setInputCol("features")
        .setOutputCol("bfeatures")

    val bucketedDf = bucketizer.transform(bucketDf)
    bucketedDf.show()
}
