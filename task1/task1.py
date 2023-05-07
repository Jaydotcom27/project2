from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, monotonically_increasing_id
import pyspark.sql.functions as F
import pyspark.sql.types as T
import sys

def main(spark, train_data, test_data):
    train_data = spark.read.csv(train_data, header=True)
    test_data = spark.read.csv(test_data, header=True)

    # columns to be outputted
    output_columns = [i for i in train_data.columns if i not in ["id", "comment_text"]]

    # Breaking down the "comment_text" column
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    train_data = train_data.withColumn("toxic", col("toxic").cast("int")).withColumn("severe_toxic", col("severe_toxic").cast("int")).withColumn("obscene", col("obscene").cast("int")).withColumn("threat", col("threat").cast("int")).withColumn("insult", col("insult").cast("int")).withColumn("identity_hate", col("identity_hate").cast("int"))
    train_data = train_data.na.drop("any")
    words_data = tokenizer.transform(train_data)

    # removing incomplete rows -> these are not allowed when using the IDF transformer so it'll break if not removed
    words_data = words_data.na.drop("any")

    # apply hashing trick and IDF transformation to tokenized text, this is similar to embedding 
    hashing_tf = HashingTF(inputCol="words", outputCol = "R")
    hash_tf = hashing_tf.transform(words_data)
    idf = IDF(inputCol = "R", outputCol = "features")
    idf_model = idf.fit(hash_tf)
    tf_idf = idf_model.transform(hash_tf)

    # preprocessing our test data
    test_data = test_data.filter(col("comment_text") != '"').withColumn('UID', monotonically_increasing_id())
    test_words = tokenizer.transform(test_data)
    test_hash_tf = hashing_tf.transform(test_words)
    test_tf_idf = idf_model.transform(test_hash_tf)
    test_results = test_data.select("UID")

    # train logistic regression model and predict
    reg_param = 0.1
    for the_column in output_columns:
        lr = LogisticRegression(featuresCol="features", labelCol=the_column, regParam=reg_param)
        lr_model = lr.fit(tf_idf.limit(4000))
        res_test = lr_model.transform(test_tf_idf)
        extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
        res_test = res_test.withColumn("proba" + '_' + the_column, extract_prob("probability")).withColumn("pred" + '_' + the_column, col('prediction'))
        res_test = res_test.select(res_test["UID"], res_test["proba" + '_' + the_column], res_test["pred" + '_' + the_column])
        test_results = test_results.join(res_test, on=["UID"])

    # print predictions
    test_results.sample(False, 0.4).show()

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Toxic Comment Classification").enableHiveSupport().getOrCreate()

    train_data = sys.argv[1]

    test_data = sys.argv[2]

    main(spark, train_data, test_data)