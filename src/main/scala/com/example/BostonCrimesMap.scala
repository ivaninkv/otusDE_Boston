package com.example

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SaveMode, SparkSession}

object BostonCrimesMap extends App {
  val spark = SparkSession.builder()
    .config("spark.sql.autoBroadcastJoinThreshold", 0)
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  val sc = spark.sparkContext
  val crimePath = args(0)
  val offenseCodesPath = args(1)
  val parquetPath = args(2)

  // read data
  val crimeFacts = spark.read.option("header", "true").option("inferSchema", "true").csv(crimePath)
    .na.fill("NA", Seq("DISTRICT"))
  val offenseCodes = spark.read.option("header", "true").option("inferSchema", "true").csv(offenseCodesPath)
    .select(col("CODE"),
      substring_index(col("NAME"), " ", 1).as("SHORT_NAME"))

  // join tables
  val offenseCodesBR = broadcast(offenseCodes)
  val df = crimeFacts.join(offenseCodesBR, crimeFacts("OFFENSE_CODE") === offenseCodes("CODE"))
    .select($"INCIDENT_NUMBER", $"DISTRICT", $"YEAR", $"MONTH", $"Lat", $"Long", $"SHORT_NAME")

  // basic stats
  val basicStats = df
    .groupBy($"DISTRICT")
    .agg(
      count("INCIDENT_NUMBER").alias("crimes_total"),
      avg("Lat").alias("lat"),
      avg("Long").alias("lng")
    )

  // monthly crimes
  df.groupBy($"DISTRICT", $"YEAR", $"MONTH")
    .count()
    .createOrReplaceTempView("vMonthCrime")
  val monthlyCrimes = spark.sql("select DISTRICT, percentile_approx(count, 0.5) as crimes_monthly " +
    "from vMonthCrime group by DISTRICT")

  // top 3 crime types
  val crimeTypes = df
    .groupBy($"DISTRICT", $"SHORT_NAME")
    .agg(count("INCIDENT_NUMBER").alias("cnt"))
    .withColumn("rn", row_number().over(Window.partitionBy($"DISTRICT")
      .orderBy($"cnt".desc)))
    .filter($"rn" <= 3)
    .drop($"rn")
    .groupBy($"DISTRICT")
    .agg(collect_list($"SHORT_NAME").alias("sh_n"))
    .withColumn("frequent_crime_types", array_join($"sh_n", ", "))
    .drop($"sh_n")

  // save data
  basicStats.join(monthlyCrimes, Seq("DISTRICT"))
    .join(crimeTypes, Seq("DISTRICT"))
    .select($"DISTRICT", $"crimes_total", $"crimes_monthly", $"frequent_crime_types", $"lat", $"lng")
    .repartition(1)
    .write
    .mode(SaveMode.Overwrite)
    .parquet(parquetPath)
}
