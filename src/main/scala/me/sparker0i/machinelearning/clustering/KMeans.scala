package me.sparker0i.machinelearning.clustering

import org.apache.{spark => sp}
import me.sparker0i.machinelearning._
import sp.ml.linalg.Vectors
import sp.ml.feature.VectorAssembler
import sp.ml.clustering.KMeans

object KMeans extends App {
    val df = spark.read
        .format("csv")
        .option("inferschema", true)
        .option("header", true)
        .load("C:\\Users\\Spark\\clustering_dataset.csv")

    df.show()

    val vectorAssembler = new VectorAssembler()
        .setInputCols(Array("col1", "col2", "col3"))
        .setOutputCol("features")

    val vectorizedDf = vectorAssembler.transform(df)
    vectorizedDf.show()

    val kMeans = new KMeans()
        .setK(3)
        .setSeed(1)

    val kModel = kMeans.fit(vectorizedDf)
    val centers = kModel.clusterCenters
    centers.foreach(println)
}
