import java.io.{File, FileWriter, PrintWriter}
import java.util.{Properties, UUID}
import java.util.Locale

import org.apache.kafka.clients.consumer.KafkaConsumer
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.kafka.common.TopicPartition
import org.apache.kafka.common.serialization.{StringDeserializer, StringSerializer}
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka010._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.Try
import scala.util.matching.Regex

object KafkaConsumerRF {
  private case class RoutingPolicy(
      anomalyThreshold: Double = 0.8,
      normalThreshold: Double = 0.2,
      anomalyTopic: String = "anomalies3",
      normalTopic: String = "normal_data",
      uncertainTopic: String = "uncertain_data"
  )

  private case class ScoredRecord(score: Double, eventTsMs: Long, label: Int)

  private lazy val MetricsFile: File =
    new File(envOrDefault("LIVE_METRICS_PATH", "SparkKafkaStreaming/results/live_metrics/live_kafka_metrics.jsonl"))

  private def metricsDir: File = MetricsFile.getParentFile

  private def envOrDefault(name: String, default: String): String =
    sys.env.getOrElse(name, default)

  private def escapeJson(value: String): String =
    value
      .replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")

  private def numericFieldPattern(field: String): Regex =
    ("\"" + field + "\"\\s*:\\s*([-0-9.Ee]+)").r

  private def stringFieldPattern(field: String): Regex =
    ("\"" + field + "\"\\s*:\\s*\"([^\"]*)\"").r

  private def extractDouble(field: String, payload: String): Option[Double] =
    numericFieldPattern(field)
      .findFirstMatchIn(payload)
      .flatMap(m => Try(m.group(1).toDouble).toOption)

  private def extractString(field: String, payload: String): Option[String] =
    stringFieldPattern(field).findFirstMatchIn(payload).map(_.group(1))

  private def percentile95(values: Seq[Long]): Long = {
    if (values.isEmpty) {
      0L
    } else {
      val sorted = values.sorted
      val index = math.ceil(sorted.size * 0.95).toInt - 1
      sorted(math.max(0, math.min(index, sorted.size - 1)))
    }
  }

  private def appendMetricsLine(line: String): Unit = this.synchronized {
    val dir = metricsDir
    if (dir != null && !dir.exists()) {
      dir.mkdirs()
    }
    val writer = new PrintWriter(new FileWriter(MetricsFile, true))
    try {
      writer.println(line)
    } finally {
      writer.close()
    }
  }

  private def formatDecimal(value: Double, scale: Int): String =
    String.format(Locale.US, s"%.${scale}f", Double.box(value))

  private def buildProducer(bootstrapServers: String): KafkaProducer[String, String] = {
    val props = new Properties()
    props.put("bootstrap.servers", bootstrapServers)
    props.put("key.serializer", classOf[StringSerializer].getName)
    props.put("value.serializer", classOf[StringSerializer].getName)
    props.put("acks", "all")
    props.put("linger.ms", "0")
    new KafkaProducer[String, String](props)
  }

  private def buildLagConsumer(bootstrapServers: String): KafkaConsumer[String, String] = {
    val props = new Properties()
    props.put("bootstrap.servers", bootstrapServers)
    props.put("group.id", s"lag-probe-${UUID.randomUUID().toString}")
    props.put("key.deserializer", classOf[StringDeserializer].getName)
    props.put("value.deserializer", classOf[StringDeserializer].getName)
    props.put("auto.offset.reset", "latest")
    props.put("enable.auto.commit", "false")
    new KafkaConsumer[String, String](props)
  }

  private def computeLag(
      bootstrapServers: String,
      offsetRanges: Array[OffsetRange]
  ): (Long, Map[String, Long]) = {
    if (offsetRanges.isEmpty) {
      return (0L, Map.empty)
    }

    val lagConsumer = buildLagConsumer(bootstrapServers)
    try {
      val partitions = offsetRanges.map(or => new TopicPartition(or.topic, or.partition)).toList.asJava
      lagConsumer.assign(partitions)
      lagConsumer.seekToEnd(partitions)

      val lagMap = offsetRanges.map { range =>
        val partition = new TopicPartition(range.topic, range.partition)
        val endOffset = lagConsumer.position(partition)
        val lag = math.max(0L, endOffset - range.untilOffset)
        s"${range.topic}-${range.partition}" -> lag
      }.toMap
      (lagMap.values.sum, lagMap)
    } finally {
      lagConsumer.close()
    }
  }

  private def routeTopic(score: Double, policy: RoutingPolicy): String = {
    if (score > policy.anomalyThreshold) {
      policy.anomalyTopic
    } else if (score <= policy.normalThreshold) {
      policy.normalTopic
    } else {
      policy.uncertainTopic
    }
  }

  private def asLong(value: Any): Long = value match {
    case null                     => 0L
    case n: java.lang.Number      => n.longValue()
    case s: String                => Try(s.toDouble.toLong).getOrElse(0L)
    case other                    => Try(other.toString.toDouble.toLong).getOrElse(0L)
  }

  private def asInt(value: Any): Int = value match {
    case null                     => -1
    case n: java.lang.Number      => n.intValue()
    case s: String                => Try(s.toDouble.toInt).getOrElse(-1)
    case other                    => Try(other.toString.toDouble.toInt).getOrElse(-1)
  }

  private def loadScores(
      spark: SparkSession,
      rdd: org.apache.spark.rdd.RDD[String],
      model: PipelineModel
  ): Map[String, ScoredRecord] = {
    import spark.implicits._

    val positiveClassProbability = udf { vector: Vector =>
      if (vector != null && vector.size > 1) vector(1) else 0.0
    }

    val scoredRows = model.transform(spark.read.json(rdd))
      .select(
        col("message_id"),
        col("event_ts_ms"),
        col("label"),
        positiveClassProbability(col("probability")).as("prediction_score")
      )
      .collect()

    scoredRows.flatMap { row =>
      Option(row.getAs[String]("message_id")).map { messageId =>
        val eventTsMs = asLong(row.getAs[Any]("event_ts_ms"))
        val label = asInt(row.getAs[Any]("label"))
        val score = row.getAs[Double]("prediction_score")
        messageId -> ScoredRecord(score = score, eventTsMs = eventTsMs, label = label)
      }
    }.toMap
  }

  def main(args: Array[String]): Unit = {
    val bootstrapServers = envOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    val inputTopic = envOrDefault("KAFKA_INPUT_TOPIC", "model-input")
    val batchIntervalSeconds = envOrDefault("STREAM_BATCH_INTERVAL_SEC", "10").toInt
    val modelPath = envOrDefault(
      "RF_MODEL_PATH",
      "SparkKafkaStreaming/model_artifacts/random_forest_smote_standard_pipeline"
    )
    val policy = RoutingPolicy(
      anomalyThreshold = envOrDefault("ANOMALY_THRESHOLD", "0.8").toDouble,
      normalThreshold = envOrDefault("NORMAL_THRESHOLD", "0.2").toDouble,
      anomalyTopic = envOrDefault("ANOMALY_TOPIC", "anomalies3"),
      normalTopic = envOrDefault("NORMAL_TOPIC", "normal_data"),
      uncertainTopic = envOrDefault("UNCERTAIN_TOPIC", "uncertain_data")
    )

    val conf = new SparkConf().setAppName("KafkaConsumerRF").setMaster(envOrDefault("SPARK_MASTER", "local[*]"))
    val ssc = new StreamingContext(conf, Seconds(batchIntervalSeconds))
    val spark = SparkSession.builder.config(ssc.sparkContext.getConf).getOrCreate()
    val model = PipelineModel.load(modelPath)

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> bootstrapServers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> envOrDefault("KAFKA_CONSUMER_GROUP", "rf-group"),
      "auto.offset.reset" -> envOrDefault("KAFKA_AUTO_OFFSET_RESET", "latest"),
      "enable.auto.commit" -> java.lang.Boolean.FALSE
    )

    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](Array(inputTopic), kafkaParams)
    )

    stream.foreachRDD { rdd =>
      if (!rdd.isEmpty()) {
        val batchStartMs = System.currentTimeMillis()
        val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
        val (totalLag, lagByPartition) = computeLag(bootstrapServers, offsetRanges)
        val scores = loadScores(spark, rdd.map(_.value()), model)
        val scoresBroadcast = rdd.sparkContext.broadcast(scores)

        val partitionStats = try {
          rdd.mapPartitions { records =>
            val producer = buildProducer(bootstrapServers)
            val latencies = ArrayBuffer.empty[Long]
            var processed = 0L
            var anomalyCount = 0L
            var normalCount = 0L
            var uncertainCount = 0L

            try {
              records.foreach { record =>
                val payload = record.value()
                val messageId = extractString("message_id", payload).getOrElse(UUID.randomUUID().toString)
                scoresBroadcast.value.get(messageId).foreach { scored =>
                  val processedTsMs = System.currentTimeMillis()
                  val latencyMs = math.max(0L, processedTsMs - scored.eventTsMs)
                  val route = routeTopic(scored.score, policy)
                  val lagValue = lagByPartition.getOrElse(s"${record.topic()}-${record.partition()}", 0L)

                  val enrichedMessage =
                    s"""{"message_id":"${escapeJson(messageId)}","event_ts_ms":${scored.eventTsMs},"processed_ts_ms":$processedTsMs,"e2e_latency_ms":$latencyMs,"prediction":${scored.score},"label":${scored.label},"route":"${escapeJson(route)}","source_topic":"${escapeJson(record.topic())}","source_partition":${record.partition()},"source_offset":${record.offset()},"consumer_lag":$lagValue}"""

                  producer.send(new ProducerRecord[String, String](route, messageId, enrichedMessage))

                  processed += 1
                  latencies += latencyMs
                  if (route == policy.anomalyTopic) {
                    anomalyCount += 1
                  } else if (route == policy.normalTopic) {
                    normalCount += 1
                  } else {
                    uncertainCount += 1
                  }
                }
              }
              producer.flush()
            } finally {
              producer.close()
            }

            Iterator((processed, anomalyCount, normalCount, uncertainCount, latencies.toList))
          }.collect()
        } finally {
          scoresBroadcast.destroy()
        }

        val totalProcessed = partitionStats.map(_._1).sum
        val anomalyCount = partitionStats.map(_._2).sum
        val normalCount = partitionStats.map(_._3).sum
        val uncertainCount = partitionStats.map(_._4).sum
        val latencies = partitionStats.flatMap(_._5)
        val meanLatencyMs = if (latencies.nonEmpty) latencies.sum.toDouble / latencies.size else 0.0
        val p95LatencyMs = percentile95(latencies)
        val batchDurationMs = System.currentTimeMillis() - batchStartMs

        val metricsLine =
          s"""{"batch_start_ms":$batchStartMs,"batch_duration_ms":$batchDurationMs,"records_processed":$totalProcessed,"anomaly_count":$anomalyCount,"normal_count":$normalCount,"uncertain_count":$uncertainCount,"mean_e2e_latency_ms":${formatDecimal(meanLatencyMs, 4)},"p95_e2e_latency_ms":$p95LatencyMs,"total_consumer_lag":$totalLag,"lag_by_partition":"${escapeJson(lagByPartition.toSeq.sortBy(_._1).mkString(","))}","routing_policy":"route_uncertain","scoring_model":"random_forest_probability"}"""

        appendMetricsLine(metricsLine)
        println(
          s"[LIVE-KAFKA-RF] processed=$totalProcessed anomaly=$anomalyCount normal=$normalCount uncertain=$uncertainCount meanLatencyMs=${formatDecimal(meanLatencyMs, 2)} p95LatencyMs=$p95LatencyMs totalLag=$totalLag batchDurationMs=$batchDurationMs"
        )

        stream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
      }
    }

    println(
      s"KafkaConsumerRF started with inputTopic=$inputTopic anomalyTopic=${policy.anomalyTopic} normalTopic=${policy.normalTopic} uncertainTopic=${policy.uncertainTopic} batchIntervalSec=$batchIntervalSeconds modelPath=$modelPath"
    )
    ssc.start()
    ssc.awaitTermination()
  }
}
