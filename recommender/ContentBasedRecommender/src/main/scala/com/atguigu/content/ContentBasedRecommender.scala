package com.atguigu.content

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jblas.DoubleMatrix

object ContentBasedRecommender {

  val MONGODB_PRODUCT_COLLECTION = "Products"
  val PRODUCT_RECS = "ContentBasedProductRecs"

  def main(args: Array[String]): Unit = {


    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://hadoop110:27017/recommender",
      "mongo.db" -> "reommender"
    )

    //创建一个SparkConf配置
    val sparkConf = new SparkConf().setAppName("ContentBasedRecommender").setMaster(config("spark.cores")).set("spark.executor.memory", "6G").set("spark.driver.memory", "2G")

    //基于SparkConf创建一个SparkSession
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    //创建一个MongoDBConfig
    val mongoConfig = MongoConfig(config("mongo.uri"), config("mongo.db"))

    import spark.implicits._

    val productRDD: RDD[(Int, String, String)] = spark.read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_PRODUCT_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product]
      .rdd
      .map(x => (x.productId, x.name, x.tags.map(c => if (c == '|') ' ' else c)))


    val productSeq: Array[(Int, String, String)] = productRDD.collect()

    val tagsData: DataFrame = spark.createDataFrame(productSeq).toDF("productId", "name", "tags")

    //实例化一个分词器默认按空格分
    // doc: 不好看(10)|送货速度快(3)|很好(100)|质量很好(1)  -> 商品详情
    // tf: 不好看的tf值: 10 / 114
    val tokenizer: Tokenizer = new Tokenizer().setInputCol("tags").setOutputCol("words")

    // 用分词器做转换，生成列“words”，返回一个dataframe，增加一列words
    val wordsData: DataFrame = tokenizer.transform(tagsData)

    wordsData.show(5)

    // HashingTF是一个工具，可以把一个词语序列，转换成词频(初始特征)
    val hashingTF: HashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(189)

    //用HashingTF做处理，返回dataframe
    val featurizedData: DataFrame = hashingTF.transform(wordsData)

    //IDF 也是一个工具，用于计算文档的IDF
    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    //将次品数据传入，得到idf模型（统计文档）
    val idfModel: IDFModel = idf.fit(featurizedData)

    //模型对原始数据做处理，计算出idf后，用tf-idf得到新的特征矩阵
    val rescaledData: DataFrame = idfModel.transform(featurizedData)

    rescaledData.show(5)

    val productFeatures: RDD[(Int, DoubleMatrix)] = rescaledData.map {
      case row => {
        (row.getAs[Int]("productId"), row.getAs[SparseVector]("features").toArray)
      }
    }
      .rdd
      .map(x => {
        (x._1, new DoubleMatrix(x._2))
      })

    val productRecs: DataFrame = productFeatures.cartesian(productFeatures)
      .filter { case (a, b) => a._1 != b._1 }
      .map {
        case (a, b) => {
          val simScore: Double = this.cosinSim(a._2, b._2)
          (a._1, (b._1, simScore))
        }
      }
      .groupByKey()
      .map {
        case (productId, items) => ProductRecs(productId, items.toList.sortWith(_._2 > _._2).map(x => Recommendation(x._1, x._2)).take(5))
      }
      .toDF()

    productRecs.show(5)

    productRecs
      .write
      .option("uri", mongoConfig.uri)
      .option("collection", PRODUCT_RECS)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()


    spark.close()

  }

  def cosinSim(product1: DoubleMatrix, product2: DoubleMatrix): Double = {
    product1.dot(product2) / (product1.norm2() * product2.norm2())
  }


}

case class MongoConfig(uri: String, db: String)

case class Product(productId: Int, name: String, categories: String, imageUrl: String, tags: String)

//推荐
case class Recommendation(rid: Int, r: Double)

//商品的相似度
case class ProductRecs(productId: Int, recs: Seq[Recommendation])
