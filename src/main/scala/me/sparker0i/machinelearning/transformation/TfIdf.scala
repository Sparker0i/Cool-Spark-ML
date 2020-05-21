package me.sparker0i.machinelearning.transformation

import me.sparker0i.machinelearning.spark
import org.apache.spark.ml.feature.{Tokenizer, HashingTF, IDF}

object TfIdf extends App {
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

    val hashingTF = new HashingTF()
        .setInputCol("words")
        .setOutputCol("rawFeatures")
        .setNumFeatures(20)

    val sentenceHashingFunctionTermFrequencyDf = hashingTF.transform(sentenceTokenizedDf)
    sentenceHashingFunctionTermFrequencyDf.show()

    val idf = new IDF()
        .setInputCol("rawFeatures")
        .setOutputCol("idfFeatures")

    val idfModel = idf.fit(sentenceHashingFunctionTermFrequencyDf)
    val tfIdfDf = idfModel.transform(sentenceHashingFunctionTermFrequencyDf)

    /* Each word relative to how frequently they appear in the corpus */
    tfIdfDf.show()
}
