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
import org.apache.spark.ml.classification.RandomForestClassifier

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
    "spark.driver.maxResultSize" -> "2g"))

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

  // Extras configuration
  // - change FILTER_OUTLINER_GOAL to true if you want to filter them
  // - change EXTRA_LINEAR_GRID to true if you want to perform a grid search on more variables for the LinearRegression
  // - change RANDOM_FOREST to true if you want to perform a random forest and a grid search on it

  val FILTER_OUTLINER_GOAL = false
  val EXTRA_LINEAR_GRID = false
  val RANDOM_FOREST = false

  // Loading data from prepro
  val datasetNoFiltered = spark.read.parquet("./resources/prepro/")

  // As var to be able to change it
  var dataset = datasetNoFiltered

  if (FILTER_OUTLINER_GOAL) {
    // Goal have most of these value between 1 and 70000. Let's remove the last 5% and the 0 values
    val Array(q95) = datasetNoFiltered.stat.approxQuantile("goal", Array(0.95), 0)
    // Remove too high value and 0 value
    dataset = datasetNoFiltered.filter($"goal" > 0).filter($"goal" < q95)
  }

  // Creation du pipeline

  // Tokenizer et stop word
  val tokenizer = new RegexTokenizer().setPattern("\\W+").setGaps(true).setInputCol("text").setOutputCol("tokens")
  val stopWordRemover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("filtered")

  // On compte les occurences du dictionnaire
  // Pas de taille pré-definie de dictionnaire
  // Pas de nombre d'occurence minimale ni maximale
  val countVectorizer = new CountVectorizer().setInputCol(stopWordRemover.getOutputCol).setOutputCol("TF")

  // IDF
  val IDF = new IDF().setInputCol(countVectorizer.getOutputCol).setOutputCol("tfidf")

  // StringIndexer: on skip les données que l'on ne connait pas (pb de l'Allemagne)
  val indexerCountry = new StringIndexer().setInputCol("country2").setOutputCol("country2_indexed").setHandleInvalid("skip")
  val indexerCurrency = new StringIndexer().setInputCol("currency2").setOutputCol("currency2_indexed")

  val oneHotEncorderCountry = new OneHotEncoderEstimator().setDropLast(false).setInputCols(Array(indexerCountry.getOutputCol, indexerCurrency.getOutputCol))
    .setOutputCols(Array("country_onehot", "currency_onehot"))

  // On assemble les vectors ensembles
  val vectorAssembler = new VectorAssembler().setInputCols(Array(
    "goal", "days_campaign", "hours_prepa", IDF.getOutputCol,
    "tfidf", "goal") ++ oneHotEncorderCountry.getOutputCols).setOutputCol("features")

  // Creation d'une foret aléatoire
  val rf = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("final_status").setPredictionCol("predictions").setImpurity("entropy")

  // Regression logisitique
  val lr = new LogisticRegression()
    .setElasticNetParam(0.4)
    .setFitIntercept(true)
    .setFeaturesCol("features")
    .setLabelCol("final_status")
    .setStandardization(true)
    .setPredictionCol("predictions")
    .setRawPredictionCol("raw_predictions")
    .setThresholds(Array(0.7, 0.3))
    .setTol(1.0e-6)
    .setMaxIter(100)

  val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, stopWordRemover, countVectorizer, IDF, indexerCountry, indexerCurrency, oneHotEncorderCountry, vectorAssembler))

  if (RANDOM_FOREST) {
    // On ajoute la random forest au pipeline
    pipeline.setStages(pipeline.getStages ++ Array(rf))

  } else {

    // On ajoute la regression  au pipeline
    pipeline.setStages(pipeline.getStages ++ Array(lr))

  }

  // Decoupage des jeux d'entrainement et test
  val Array(training, test) = dataset.randomSplit(Array(0.9, 0.1), 1234)

  // Fit des données
  val model = pipeline.fit(training)

  // Prédiction
  val dfWithSimplePredictions = model.transform(test)

  /**
   * Affiche de résultats:
   *  - tableau des valeurs
   *  - calcul du F1, de la précision et du recall
   */
  dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()
  val f1Evaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("f1")
  var f1 = f1Evaluator.evaluate(dfWithSimplePredictions)

  val precisionEvaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("weightedPrecision")
  var precision = precisionEvaluator.evaluate(dfWithSimplePredictions)

  val recallEvaluator = new MulticlassClassificationEvaluator().setLabelCol("final_status").setPredictionCol("predictions").
    setMetricName("weightedRecall")
  var recall = recallEvaluator.evaluate(dfWithSimplePredictions)

  if (RANDOM_FOREST)
    println("Les resultats pour le modèle \" Random Forest\" sont:");
  else
    println("Les resultats pour le modèle \"Linear Regression\" sont:");
  println(f"F1 precision = $f1%1.4f")
  println(f"Recall       = $recall%1.4f")
  println(f"Precision    = $precision%1.4f")

  println("\nNous lançons la recherche sur une grille de valeur\n ")

  // Default value for the grid
  var paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(1e-8, 1e-6, 1e-4, 1e-2)) //, 0.1, 0.5, 1, 2))
    .addGrid(countVectorizer.minDF, (55.0 to 95.0 by 20).toArray)
    .build()

  if (EXTRA_LINEAR_GRID) {
    paramGrid = new ParamGridBuilder().addGrid(lr.elasticNetParam, (0.2 to 1.0 by 0.1).toArray)
      .addGrid(lr.regParam, Array(1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1, 2))
      .addGrid(countVectorizer.minDF, (10.0 to 95.0 by 20).toArray)
      .build()

  } else if (RANDOM_FOREST)
    paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, (2 to 15 by 1).toArray)
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .addGrid(rf.numTrees, (50 to 1000 by 150).toArray)
      .addGrid(countVectorizer.minDF, (10.0 to 95.0 by 20).toArray).build()

  val trainValidationSplit = new TrainValidationSplit().setEstimator(pipeline).setEvaluator(f1Evaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.7)

  val trainSplit = trainValidationSplit.fit(training)
  val testTransformed = trainSplit.transform(test)

  println("Resultat du meilleur modele")
  testTransformed.groupBy("final_status", "predictions").count.show()

  f1 = f1Evaluator.evaluate(testTransformed)
  recall = recallEvaluator.evaluate(testTransformed)
  precision = precisionEvaluator.evaluate(testTransformed)
  println("Les resultats pour le meilleur modèle de la grille sont:");
  println(f"F1 precision = $f1%1.4f")
  println(f"Recall       = $recall%1.4f")
  println(f"Precision    = $precision%1.4f")

  println("Les valeurs des paramètres pour ce modèle sont les suivants:")
  println(trainSplit.getEstimatorParamMaps(trainSplit.validationMetrics.indexOf(trainSplit.validationMetrics.max)))

  // Saving the modele
  trainSplit.write.save("./resources/trained/")
}
