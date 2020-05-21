package me.sparker0i.machinelearning.transformation

import me.sparker0i.machinelearning._
import org.apache.spark.ml.feature.Tokenizer

object Tokenize extends App {
    val sentencesDf = spark.createDataFrame(
        Array(
            (1, "This is an introduction to Spark ML"),
            (2, "MLLib includes libraries for classification and regression"),
            (3, "It also contains supporting tools for pipelines")
        )
    ).toDF("id", "sentence")

    sentencesDf.show()

    val sentenceToken = new Tokenizer()
        .setInputCol("sentence")
        .setOutputCol("words")

    val sentenceTokenizedDf = sentenceToken.transform(sentencesDf)
    sentenceTokenizedDf.show()
}
