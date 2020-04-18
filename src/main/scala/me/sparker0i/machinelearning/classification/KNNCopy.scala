package me.sparker0i.machinelearning.classification

import me.sparker0i.machinelearning._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable

object KNNCopy {
    import spark.implicits._
    var df: DataFrame = _

    def main(args: Array[String]): Unit = {
        System.setProperty("hadoop.home.dir", "C:\\spark\\")
        Logger.getLogger("org").setLevel(Level.ERROR)

        df = readCsv(fileName = "C:\\Users\\Spark\\Downloads\\iris.csv", header = true)

        val indexer = new StringIndexer()
            .setInputCol("Class")
            .setOutputCol("Class_Indexed")

        df = indexer.fit(df).transform(df)

        val classValues = df.select($"Class", $"Class_Indexed")
            .distinct()

        df = df.drop($"Class")

        val nFolds = 5
        val k = 5

        df.show()
        classValues.show()

        val folds = df.randomSplit(Array.fill(nFolds)(1.0/nFolds))
        var scores = mutable.MutableList[Double]()

        for (i <- folds.indices) {
            var ts = folds
            val testSet = ts(i).collect()
            ts = ts.zipWithIndex.filter(_._2 != i)
                .map(_._1)

            var tSS = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], testSet(0).schema)

            for (j <- ts.indices) {
                tSS = tSS.union(ts(j))
                println(s"U ${tSS.count()}")
            }

            val trainSet = tSS.collect()

            val predictions = mutable.MutableList[Double]()
            testSet.foreach{row =>
                var distances = mutable.MutableList[(Row, Double)]()
                trainSet.foreach{trainRow =>
                    var distance = 0.0
                    for (i <- trainRow.schema.fields.indices) {
                        distance += math.pow(trainRow(i).asInstanceOf[Double] - row(i).asInstanceOf[Double], 2)
                    }

                    val dist = math.sqrt(distance)
                    val x = (trainRow, dist)
                    distances += x
                }

                distances = distances.sortBy(_._2)
                var neighbours = mutable.MutableList[Row]()

                for (i <- 1 to k) {
                    neighbours += distances(i)._1
                }
                neighbours.toList

                val outputValues = for (row <- neighbours) yield row(trainSet(0).length - 1).asInstanceOf[Double]
                val output = outputValues.groupBy(identity)
                    .mapValues(_.size)
                    .toSeq
                    .sortBy(_._1)
                    .sortWith(_._2 > _._2)
                    .head._1

                predictions += output
            }
            val predicted = predictions.toList
            val actual = for (row <- testSet) yield row(testSet(0).length - 1).asInstanceOf[Double]

            scores += accuracyMetric(actual.toList, predicted)
        }
        println(scores)
        println(s"Accuracy : ${scores.sum/scores.length}%")
    }

    def readCsv(fileName: String, header: Boolean): DataFrame = {
        spark.read
            .format("csv")
            .option("header", header)
            .option("inferSchema", header)
            .load(fileName)
            .repartition($"Class")
    }

    def accuracyMetric(actual: List[Double], predicted: List[Double]): Double = {
        var correct = 0.0
        for (i <- actual.indices) {
            if (actual(i) == predicted(i))
                correct += 1
        }
        correct / actual.length * 100
    }
}
