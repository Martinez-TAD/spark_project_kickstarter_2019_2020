package paristech
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import scala.annotation.meta.param
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.functions._

object Trainer extends App {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  val conf = new SparkConf().setAll(Map(
    "spark.scheduler.mode" -> "FIFO",
    "spark.speculation" -> "false",
    "spark.reducer.maxSizeInFlight" -> "48m",
    "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max" -> "1g",
    "spark.shuffle.file.buffer" -> "32k",
    "spark.default.parallelism" -> "12",
    "spark.sql.shuffle.partitions" -> "12",
    "spark.driver.maxResultSize" -> "2g",
    "spark.master" -> "local[*]"))

  val spark = SparkSession
    .builder
    .config(conf)
    .appName("TP Spark : Trainer")
    .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    
  import spark.implicits._
/*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

  val datasetLoaded = spark.read.parquet("/home/martinez/spark-project/data/parquet2/")
  
  val datasetNoFiltered = datasetLoaded.withColumn("goal", $"goal".cast("Double"))
  
  datasetNoFiltered.printSchema()
  val Array(q95) = datasetNoFiltered.stat.approxQuantile("goal", Array(0.95),0)


  
  // Remove too high value and 0 value
  val dataset = datasetNoFiltered.filter($"goal">0).filter($"goal"<q95 )
    
  // Creation du pipeline

  val tokenizer = new RegexTokenizer().setPattern("\\W+").setGaps(true).setInputCol("text").setOutputCol("tokens")
  val stopWordRemover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")

  // On compte les occurences du dictionnaire
  // Pas de taille pré-definie de dictionnaire
  // Pas de nombre d'occurence minimale ni maximale
  val countVectorizer = new CountVectorizer().setInputCol(stopWordRemover.getOutputCol).setOutputCol("TF").setMinDF(55)

  //    val dataSetHashing = new HashingTF().setInputCol("filtered").setOutputCol("features").transform(dataSetCountVect)
  //
  val IDF = new IDF().setInputCol(countVectorizer.getOutputCol).setOutputCol("tfidf")

  //  val dataSetTFIDF = dataSetIDF.withColumn("TFIDF", $"TF"*$"IDF")

  val indexerCountry = new StringIndexer().setInputCol("country2").setOutputCol("country2_indexed").setHandleInvalid("skip")
  val indexerCurrency = new StringIndexer().setInputCol("currency2").setOutputCol("currency2_indexed")

  val oneHotEncorderCountry = new OneHotEncoderEstimator().setDropLast(false).setInputCols(Array(indexerCountry.getOutputCol, indexerCurrency.getOutputCol))
    .setOutputCols(Array("country_onehot", "currency_onehot"))
    
  val vectorAssembler = new VectorAssembler().setInputCols(Array("tfidf", "goal", "backers_count","days_campaign", "hours_prepa", "country_onehot", "currency_onehot")).setOutputCol("features")//.setHandleInvalid("skip")

  val lr = new LogisticRegression().setElasticNetParam(0.0)
    .setFitIntercept(true)
    .setFeaturesCol("features")
    .setLabelCol("final_status")
    .setStandardization(true)
    .setPredictionCol("predictions")
    .setRawPredictionCol("raw_predictions")
    .setThresholds(Array(0.7, 0.3))
    .setTol(1.0e-6)
    .setMaxIter(100)

  val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, stopWordRemover, countVectorizer, IDF, indexerCountry, indexerCurrency, oneHotEncorderCountry, vectorAssembler, lr))

  val Array(training, test) = dataset.randomSplit(Array(0.9, 0.1), 999)

  val model = pipeline.fit(training)
  val dfWithSimplePredictions = model.transform(test)

  dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()
  val f1Evaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("f1")
  val f1 = f1Evaluator.evaluate(dfWithSimplePredictions)

  val precisionEvaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("weightedPrecision")
    val precision = precisionEvaluator.evaluate(dfWithSimplePredictions)

  val recallEvaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("weightedRecall")
   val recall = recallEvaluator.evaluate(dfWithSimplePredictions)

  println(s"F1 precision = ${f1}")
  println(s"Recall = ${recall}")
  println(s"Precision = ${precision}")

  //dataSetIDF.select("features", "predictions", "final_status","raw_predictions").show()
  

  val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(1e-8, 1e-6, 1e-4, 1e-2))
                  .addGrid(countVectorizer.minDF, (30.0 to 1500.0 by 30.0).toArray)
                  .build()

  val trainValidationSplit = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(f1Evaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.7)

  val trainSplit = trainValidationSplit.fit(training)
  val testTransformed = trainSplit.transform(test)
  val gridf1 = f1Evaluator.evaluate(testTransformed)
  println(s"F1 precision = ${gridf1}")
  println(s"Recall = ${recallEvaluator.evaluate(testTransformed)}")
  println(s"Precision = ${precisionEvaluator.evaluate(testTransformed)}")
  
  
}