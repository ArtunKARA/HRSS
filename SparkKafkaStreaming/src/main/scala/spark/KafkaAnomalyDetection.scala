import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.sys.process._
import scala.util.Try

object KafkaAnomalyDetection {
  private def envOrDefault(name: String, default: String): String =
    sys.env.getOrElse(name, default)

  private def boolEnv(name: String, default: Boolean): Boolean =
    Try(envOrDefault(name, default.toString).toBoolean).getOrElse(default)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("KafkaAnomalyDetection")
      .master(envOrDefault("SPARK_MASTER", "local[*]"))
      .getOrCreate()

    val runScoringCommand = boolEnv("RUN_SCORING_COMMAND", default = false)
    val scoringCommand = envOrDefault("SCORING_COMMAND", "")
    val predictionsPath = envOrDefault("PREDICTIONS_PATH", "SparkKafkaStreaming/predictions.csv")
    val predictionColumn = envOrDefault("PREDICTION_COLUMN", "predictions")
    val bootstrapServers = envOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    val anomalyThreshold = envOrDefault("ANOMALY_THRESHOLD", "0.8").toDouble
    val normalThreshold = envOrDefault("NORMAL_THRESHOLD", "0.2").toDouble
    val anomalyTopic = envOrDefault("ANOMALY_TOPIC", "anomalies3")
    val normalTopic = envOrDefault("NORMAL_TOPIC", "normal_data")
    val uncertainTopic = envOrDefault("UNCERTAIN_TOPIC", "uncertain_data")
    val writeToKafka = boolEnv("WRITE_TO_KAFKA", default = true)

    if (runScoringCommand && scoringCommand.trim.nonEmpty) {
      val result = Seq("bash", "-lc", scoringCommand).!
      if (result != 0) {
        println(s"Scoring command failed with code $result")
        spark.stop()
        sys.exit(1)
      }
      println("Scoring command executed successfully.")
    }

    val predictions = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(predictionsPath)

    val anomalies = predictions
      .filter(s"$predictionColumn > $anomalyThreshold")
      .withColumn("route_topic", lit(anomalyTopic))
      .withColumn("routing_policy", lit("route_uncertain"))

    val normalData = predictions
      .filter(s"$predictionColumn <= $normalThreshold")
      .withColumn("route_topic", lit(normalTopic))
      .withColumn("routing_policy", lit("route_uncertain"))

    val uncertainData = predictions
      .filter(s"$predictionColumn > $normalThreshold AND $predictionColumn <= $anomalyThreshold")
      .withColumn("route_topic", lit(uncertainTopic))
      .withColumn("routing_policy", lit("route_uncertain"))

    val totalCount = predictions.count()
    val anomalyCount = anomalies.count()
    val normalCount = normalData.count()
    val uncertainCount = uncertainData.count()

    val anomalyRatio = if (totalCount > 0) anomalyCount.toDouble / totalCount else 0.0
    val normalRatio = if (totalCount > 0) normalCount.toDouble / totalCount else 0.0
    val uncertainRatio = if (totalCount > 0) uncertainCount.toDouble / totalCount else 0.0

    println(s"Total records: $totalCount")
    println(s"Anomaly Topic Count: $anomalyCount")
    println(s"Normal Topic Count: $normalCount")
    println(s"Uncertain Topic Count: $uncertainCount")
    println(f"Anomaly Ratio: $anomalyRatio%.4f")
    println(f"Normal Ratio: $normalRatio%.4f")
    println(f"Uncertain Ratio: $uncertainRatio%.4f")
    println(
      s"Routing policy: route_uncertain thresholds=(anomaly>$anomalyThreshold, normal<=$normalThreshold) topics=($anomalyTopic, $normalTopic, $uncertainTopic)"
    )

    def sendToKafka(df: DataFrame, topic: String): Unit = {
      val jsonDF = df.toJSON.toDF("value")
      jsonDF.write
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrapServers)
        .option("topic", topic)
        .save()
    }

    if (writeToKafka) {
      try {
        sendToKafka(anomalies, anomalyTopic)
        sendToKafka(normalData, normalTopic)
        sendToKafka(uncertainData, uncertainTopic)
        println("Records were written to Kafka with route_uncertain policy.")
      } catch {
        case ex: Throwable =>
          println(s"Kafka write skipped or failed: ${ex.getMessage}")
      }
    } else {
      println("WRITE_TO_KAFKA=false, Kafka write intentionally skipped.")
    }

    spark.stop()
  }
}
