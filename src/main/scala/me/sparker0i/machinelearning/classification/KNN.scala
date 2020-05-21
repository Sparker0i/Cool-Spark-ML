package me.sparker0i.machinelearning.classification

import me.sparker0i.machinelearning._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

object KNN {
    import spark.implicits._

    def main(args: Array[String]): Unit = {
        Logger.getLogger("org").setLevel(Level.ERROR)

        var df = readCsv(fileName = "C:\\Users\\Spark\\Downloads\\iris.csv", header = true)

        df = df.transform(normalizeData)
            .transform(moveClassToEnd)

        df.show()

        val nFolds = 5
        val folds = df.randomSplit(Array.fill(nFolds)(1.0/nFolds))

        for (k <- 1 to 10) {
            val scores = evaluateAlgorithm(folds, k, kNN)
            println(s"Scores = $scores")
            println(s"Accuracy at k = $k : ${scores.sum/scores.length} %\n")
        }
    }

    def normalizeData(df: DataFrame): DataFrame = {
        var x = df
        df.columns.filterNot(e => e == "Class").foreach{col =>
            val (mean_col, stddev_col) = df.select(mean(col), stddev(col))
                .as[(Double, Double)]
                .first()
            x = x.withColumn(s"$col.norm", ($"$col" - mean_col) / stddev_col)
                .drop(col)
                .withColumnRenamed(s"$col.norm", col)
        }
        x
    }

    def moveClassToEnd(df: DataFrame): DataFrame = {
        val cols = df.columns.filterNot(_ == "Class") ++ Array("Class")
        df.select(cols.head, cols.tail: _*)
    }

    def readCsv(fileName: String, header: Boolean): DataFrame = {
        spark.read
            .format("csv")
            .option("header", header)
            .load(fileName)
            .repartition($"Class")
    }

    def evaluateAlgorithm(folds: Array[Dataset[Row]], k: Int, algorithm: (Array[Row], Array[Row], Int) => List[String]): List[Double] = {
        val scores = for (i <- folds.indices) yield {
            val trainSet = folds.zipWithIndex.filter(_._2 != i)
                .map(_._1)
                .reduce((a, b) => a.union(b))
            val testSet = folds(i).collect()

            val predicted = algorithm(trainSet.collect(), testSet, k)
            val actual = (for (row <- testSet) yield row.getString(testSet(0).length - 1)).toList

            accuracyMetric(actual, predicted)
        }
        scores.toList
    }

    def accuracyMetric(actual: List[String], predicted: List[String]): Double = {
        var correct = 0.0
        for (i <- actual.indices) {
            if (actual(i) == predicted(i))
                correct += 1
        }
        correct / actual.length * 100
    }

    def computeEuclideanDistance(row1: Row, row2: Row): Double = {
        var distance = 0.0
        for (i <- 0 until row1.length - 1) {
            distance += math.pow(row1.getDouble(i) - row2.getDouble(i), 2)
        }
        math.sqrt(distance)
    }

    def kNN(trainSet: Array[Row], testSet: Array[Row], k: Int): List[String] = {
        (for (row <- testSet) yield predictClassification(trainSet, row, k)).toList
    }

    def predictClassification(trainSet: Array[Row], testRow: Row, k: Int): String = {
        val neighbours = getNeighbours(trainSet, testRow, k)
        (for (row <- neighbours) yield row.getString(trainSet(0).length - 1))
            .groupBy(identity)
            .mapValues(_.size)
            .toSeq
            .sortWith(_._2 > _._2)
            .head._1
    }

    def getNeighbours(trainSet: Array[Row], testRow: Row, k: Int): List[Row] = {
        val distances = (for (trainRow <- trainSet) yield (trainRow, computeEuclideanDistance(trainRow, testRow)))
            .sortBy(_._2)
        val neighbours = for (i <- 1 to k) yield distances(i)._1
        neighbours.toList
    }
}
