import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object RandomForestModelExport {
  private def envOrDefault(name: String, default: String): String =
    sys.env.getOrElse(name, default)

  private def writeMetadata(path: String, dataPath: String, featureCols: Seq[String], numTrees: Int): Unit = {
    val parent = Paths.get(path).getParent
    if (parent != null) {
      Files.createDirectories(parent)
    }
    val escapedDataPath = dataPath.replace("\\", "\\\\")
    val featureColumnsJson = featureCols
      .map { featureCol =>
        "\"" + featureCol.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
      }
      .mkString(", ")
    val json =
      s"""{
         |  "training_data_path": "$escapedDataPath",
         |  "feature_columns": [$featureColumnsJson],
         |  "num_trees": $numTrees,
         |  "label_column": "label",
         |  "features_column": "scaledFeatures"
         |}
         |""".stripMargin
    Files.write(Paths.get(path), json.getBytes(StandardCharsets.UTF_8))
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("RandomForestModelExport")
      .master(envOrDefault("SPARK_MASTER", "local[*]"))
      .getOrCreate()

    val dataPath = envOrDefault("RF_TRAINING_DATA_PATH", "Data/HRSS_SMOTE_standard.csv")
    val outputPath = envOrDefault(
      "RF_MODEL_PATH",
      "SparkKafkaStreaming/model_artifacts/random_forest_smote_standard_pipeline"
    )
    val metadataPath = envOrDefault(
      "RF_MODEL_METADATA_PATH",
      "SparkKafkaStreaming/model_artifacts/random_forest_smote_standard_pipeline_metadata.json"
    )
    val numTrees = envOrDefault("RF_NUM_TREES", "2000").toInt

    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)

    val featureCols = rawData.columns.filterNot(colName => colName == "Timestamp" || colName == "Labels")
    val trainingData = rawData.withColumn("label", col("Labels").cast("double"))

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    val randomForest = new RandomForestClassifier()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setNumTrees(numTrees)
      .setSeed(42L)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, randomForest))
    val pipelineModel = pipeline.fit(trainingData)
    pipelineModel.write.overwrite().save(outputPath)
    writeMetadata(metadataPath, dataPath, featureCols, numTrees)

    println(s"RandomForestModelExport wrote model to $outputPath")
    println(s"RandomForestModelExport wrote metadata to $metadataPath")
    println(s"RandomForestModelExport featureCount=${featureCols.length} numTrees=$numTrees dataPath=$dataPath")

    spark.stop()
  }
}
