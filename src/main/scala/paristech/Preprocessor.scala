package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.functions._

object Preprocessor extends App {
  // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
  // On vous donne un exemple de setting quand même
  val conf = new SparkConf().setAll(Map(
    "spark.scheduler.mode" -> "FIFO",
    "spark.speculation" -> "false",
    "spark.reducer.maxSizeInFlight" -> "48m",
    "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
    "spark.kryoserializer.buffer.max" -> "1g",
    "spark.shuffle.file.buffer" -> "32k",
    "spark.default.parallelism" -> "12",
    "spark.sql.shuffle.partitions" -> "12",
    "spark.master" -> "local"))

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
  // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
  val spark = SparkSession
    .builder
    .config(conf)
    .appName("TP Spark : Preprocessor")
    .getOrCreate()

  import spark.implicits._

/*******************************************************************************
			 *
			 *       TP 2
			 *
			 *       - Charger un fichier csv dans un dataFrame
			 *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
			 *       - Sauver le dataframe au format parquet
			 *
			 *       if problems with unimported modules => sbt plugins update
			 *
			 ********************************************************************************/

  val df: DataFrame = spark
    .read
    .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
    .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
    .csv("/home/martinez/git/cours-spark-telecom/data/train_clean.csv")

  println(s"Nombre de lignes : ${df.count}")
  println(s"Nombre de colonnes : ${df.columns.length}")

  df.show()

  val dfCasted: DataFrame = df
    .withColumn("goal", $"goal".cast("Int"))
    .withColumn("deadline", $"deadline".cast("Int"))
    .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
    .withColumn("created_at", $"created_at".cast("Int"))
    .withColumn("launched_at", $"launched_at".cast("Int"))
    .withColumn("backers_count", $"backers_count".cast("Int"))
    .withColumn("final_status", $"final_status".cast("Int"))

  // Droping disable_communication column
  val df2: DataFrame = dfCasted.drop("disable_communication")

  // Droping futur leaks
  val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

  // Cleaning the country/currency columns
  def cleanCountry(country: String, currency: String): String = {
    if (country == "False")
      currency
    else
      country
  }

  def cleanCurrency(currency: String): String = {
    if (currency != null && currency.length != 3)
      null
    else
      currency
  }

  val cleanCountryUdf = udf(cleanCountry _)
  val cleanCurrencyUdf = udf(cleanCurrency _)

  val dfCountry: DataFrame = dfNoFutur
    .withColumn("country2", cleanCountryUdf($"country", $"currency"))
    .withColumn("currency2", cleanCurrencyUdf($"currency"))
    .drop("country", "currency")

  val dfNoCC = dfNoFutur
    .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
    .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
    //.withColumn("country2", when(length($"country2") =!= 2, null).otherwise($"country2")) //adding to remove
    .drop("country", "currency")

  // Remove non need lines
  val filterFinalSatus = dfNoCC.filter(length($"country2") === 2).filter($"final_status" === 0 || $"final_status" === 1)

  // Adding some new columns
  val dfTS = filterFinalSatus.withColumn("deadlineTS", from_unixtime($"deadline")).withColumn("launched_atTS", from_unixtime($"launched_at")).withColumn("creadted_atTS", from_unixtime($"created_at"))
  val datedTS = dfTS.withColumn("days_campaign", datediff($"deadlineTS", $"launched_atTS")).withColumn("hours_prepa", round(($"launched_at" - $"created_at") / 3600.floatValue(), 3)).drop("deadline", "deadlineTS", "launched_at", "launched_atTS", "created_at")

  // Cleaning text
  val cleanText = datedTS.withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))
      .drop("name", "desc", "keywords")

  val finalDS = cleanText
    .withColumn("country2", when(length($"country2") =!= 2, "unknown").otherwise($"country2"))
    .withColumn("days_campaign", when($"days_campaign".isNull, "-1").otherwise($"days_campaign"))
    .withColumn("hours_prepa", when($"hours_prepa".isNull, "-1").otherwise($"hours_prepa"))
    .withColumn("goal", when($"goal".isNull, "-1").otherwise($"goal"))
    
   finalDS.write.parquet("/home/martinez/spark-project/data/parquet/")

}