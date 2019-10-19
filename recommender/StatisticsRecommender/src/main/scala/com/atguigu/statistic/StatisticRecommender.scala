package com.atguigu.statistic

import java.sql.Date
import java.text.SimpleDateFormat

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object StatisticRecommender {

  val MONGODB_RATING_COLLECTION = "Rating"
  val MONGODB_PRODUCT_COLLECTION = "Products"

  //统计表的名称
  val RATE_MORE_PRODUCTS = "RateMoreProducts"
  val RATE_MORE_RECENTLY_PRODUCTS = "RateMoreRecentlyProducts"
  val AVERAGE_PRODUCTS = "AverageProducts"

  def main(args: Array[String]): Unit = {

    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop110:27017/recommender",
      "mongo.db" -> "recommender"
    )

    // 创建SparkConf配置
    val sparkConf = new SparkConf().setAppName("StatisticRecommender").setMaster(config("spark.cores"))

    // 创建SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // 调高日志等级
    spark.sparkContext.setLogLevel("ERROR")

    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    //加入隐士转换
    import spark.implicits._

    //数据加载
    val ratingDF: DataFrame = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating]
      .toDF()

    val productDF: DataFrame = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_PRODUCT_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product]
      .toDF()

    //保存成临时表
    ratingDF.createOrReplaceTempView("ratings")

    val rateMoreProductsDF: DataFrame = spark.sql("select productId,count(productId) as count from ratings group by productId")

    rateMoreProductsDF.show(5)

    rateMoreProductsDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", RATE_MORE_PRODUCTS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    val simpleDateFormat = new SimpleDateFormat("yyyyMM")

    // todo 自定义UDF函数
    spark.udf.register("changeDate", (x: Long) => simpleDateFormat.format(new Date(x * 1000L)).toInt)

    val ratingOfYeardMonth: DataFrame = spark.sql("select productId,score,changeDate(timestamp) as yearmonth from ratings")

    ratingOfYeardMonth.createOrReplaceTempView("ratingOfMonth")

    val rateMoreRecentlyProducts: DataFrame =
      spark.sql("select productId, count(productId) as count, yearmonth from ratingOfMonth group by yearmonth, productId")

    rateMoreRecentlyProducts.show(5)

    rateMoreRecentlyProducts.write
      .option("uri", mongoConfig.uri)
      .option("collection", RATE_MORE_RECENTLY_PRODUCTS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    val averageProductsDF: DataFrame = spark.sql("select productId,avg(score) as avg from ratings group by productId order by avg desc")

    averageProductsDF.show(5)

    averageProductsDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", AVERAGE_PRODUCTS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()


    spark.stop()

  }

}


case class Product(productId: Int, name: String, categories: String, imageUrl: String, tags: String)

case class Rating(userId: Int, productId: Int, score: Double, timestamp: Long)

case class MongoConfig(uri: String, db: String)

case class Recommendation(rid: Int, r: Double)
