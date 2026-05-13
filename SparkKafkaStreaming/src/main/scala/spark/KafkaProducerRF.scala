import java.io.File
import java.util.Properties

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.kafka.common.serialization.StringSerializer

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Try

object KafkaProducerRF {
  private case class SourceConfig(path: String, requiredLabel: Int, name: String)
  private case class SourceRow(source: String, rowMap: Map[String, String])

  private def envOrDefault(name: String, default: String): String =
    sys.env.getOrElse(name, default)

  private def escapeJson(value: String): String =
    value
      .replace("\\", "\\\\")
      .replace("\"", "\\\"")
      .replace("\n", "\\n")
      .replace("\r", "\\r")

  private def parseNumeric(value: String): Option[String] = {
    val trimmed = value.trim
    if (trimmed.isEmpty) {
      None
    } else {
      Try(trimmed.toDouble).toOption.map { numeric =>
        if (numeric.isWhole()) {
          numeric.toLong.toString
        } else {
          trimmed
        }
      }
    }
  }

  private def loadRows(config: SourceConfig): (Array[String], Vector[SourceRow]) = {
    val source = Source.fromFile(config.path)
    try {
      val lines = source.getLines()
      if (!lines.hasNext) {
        throw new IllegalStateException(s"CSV file is empty: ${config.path}")
      }

      val headers = lines.next().split(",", -1).map(_.trim)
      val rows = lines.flatMap { line =>
        val values = line.split(",", -1).map(_.trim)
        val rowMap = headers.zipAll(values, "", "").toMap
        val label = parseNumeric(rowMap.getOrElse("Labels", "-1")).map(_.toInt).getOrElse(-1)
        if (label == config.requiredLabel) {
          Some(SourceRow(config.name, rowMap))
        } else {
          None
        }
      }.toVector
      (headers, rows)
    } finally {
      source.close()
    }
  }

  private def interleave(normalRows: Vector[SourceRow], anomalousRows: Vector[SourceRow]): Vector[SourceRow] = {
    val merged = ArrayBuffer.empty[SourceRow]
    val normalIterator = normalRows.iterator
    val anomalousIterator = anomalousRows.iterator
    val normalPerAnomaly = math.max(1, normalRows.size / math.max(1, anomalousRows.size))

    while (normalIterator.hasNext || anomalousIterator.hasNext) {
      var emittedNormals = 0
      while (normalIterator.hasNext && emittedNormals < normalPerAnomaly) {
        merged += normalIterator.next()
        emittedNormals += 1
      }
      if (anomalousIterator.hasNext) {
        merged += anomalousIterator.next()
      }
      if (!normalIterator.hasNext) {
        while (anomalousIterator.hasNext) {
          merged += anomalousIterator.next()
        }
      }
      if (!anomalousIterator.hasNext) {
        while (normalIterator.hasNext) {
          merged += normalIterator.next()
        }
      }
    }

    merged.toVector
  }

  def main(args: Array[String]): Unit = {
    val bootstrapServers = envOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    val inputTopic = envOrDefault("KAFKA_INPUT_TOPIC", "model-input")
    val normalPath = envOrDefault("STREAM_SOURCE_NORMAL_CSV_PATH", new File("Data/HRSS_normal_standard.csv").getPath)
    val anomalousPath = envOrDefault("STREAM_SOURCE_ANOMALOUS_CSV_PATH", new File("Data/HRSS_anomalous_standard.csv").getPath)
    val flowRateMs = envOrDefault("REPLAY_FLOW_RATE_MS", "10").toLong
    val maxMessages = envOrDefault("MAX_MESSAGES", "0").toInt

    val props = new Properties()
    props.put("bootstrap.servers", bootstrapServers)
    props.put("key.serializer", classOf[StringSerializer].getName)
    props.put("value.serializer", classOf[StringSerializer].getName)
    props.put("acks", "all")
    props.put("linger.ms", "0")

    val (normalHeaders, normalRows) = loadRows(SourceConfig(normalPath, requiredLabel = 0, name = "normal_standard_raw"))
    val (anomalousHeaders, anomalousRows) = loadRows(SourceConfig(anomalousPath, requiredLabel = 1, name = "anomalous_standard_raw"))
    if (!normalHeaders.sameElements(anomalousHeaders)) {
      throw new IllegalStateException("Normal and anomalous raw CSV headers do not match.")
    }

    val headers = normalHeaders
    val excludedHeaders = Set("Timestamp", "Labels")
    val featureHeaders = headers.filterNot(excludedHeaders.contains)
    val orderedRows = interleave(normalRows, anomalousRows)
    val producer = new KafkaProducer[String, String](props)

    try {
      var sentCount = 0
      orderedRows.iterator.takeWhile(_ => maxMessages <= 0 || sentCount < maxMessages).foreach { sourceRow =>
        val rowMap = sourceRow.rowMap
        val messageId = f"msg-${sentCount + 1}%08d"
        val eventTsMs = System.currentTimeMillis()
        val sourceTimestamp = rowMap.getOrElse("Timestamp", "0.0")
        val label = rowMap.getOrElse("Labels", "-1")

        val featurePairs = featureHeaders.map { header =>
          val value = rowMap.getOrElse(header, "")
          parseNumeric(value) match {
            case Some(numeric) => s"""\"${escapeJson(header)}\":$numeric"""
            case None          => s"""\"${escapeJson(header)}\":null"""
          }
        }

        val sourceTimestampJson = parseNumeric(sourceTimestamp).getOrElse("0.0")
        val labelJson = parseNumeric(label).getOrElse("-1")

        val payload =
          "{" +
            (
              Seq(
                s"""\"message_id\":\"${escapeJson(messageId)}\"""",
                s"""\"event_ts_ms\":$eventTsMs""",
                s"""\"source_timestamp\":$sourceTimestampJson""",
                s"""\"label\":$labelJson""",
                s"""\"source_dataset\":\"${escapeJson(sourceRow.source)}\""""
              ) ++ featurePairs
            ).mkString(",") +
            "}"

        producer.send(new ProducerRecord[String, String](inputTopic, messageId, payload))
        sentCount += 1

        if (flowRateMs > 0) {
          Thread.sleep(flowRateMs)
        }
      }

      producer.flush()
      println(
        s"KafkaProducerRF replay completed. topic=$inputTopic sent=$sentCount flowRateMs=$flowRateMs normalPath=$normalPath anomalousPath=$anomalousPath normalRows=${normalRows.size} anomalousRows=${anomalousRows.size}"
      )
    } finally {
      producer.close()
    }
  }
}
