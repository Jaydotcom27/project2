from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, monotonically_increasing_id
import pyspark.sql.functions as F
import pyspark.sql.types as T
import sys


def main():
    # create a spark session
    spark = SparkSession.builder.appName("task1").getOrCreate()

    # read training and test data from CSV files
    train_data = spark.read.csv(sys.argv[1], header=True)
    test_data = spark.read.csv(sys.argv[2], header=True)

    # list of output columns
    out_cols = [i for i in train_data.columns if i not in ["id", "comment_text"]]

    # tokenize the "comment_text" column
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    words_data = tokenizer.transform(train_data).select("id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "words")
    words_data = words_data.withColumn("toxic", col("toxic").cast("int")).withColumn("severe_toxic", col("severe_toxic").cast("int")).withColumn("obscene", col("obscene").cast("int")).withColumn("threat", col("threat").cast("int")).withColumn("insult", col("insult").cast("int")).withColumn("identity_hate", col("identity_hate").cast("int"))

    # remove any rows that contain null values
    words_data = words_data.na.drop("any")

    # apply hashing trick and IDF transformation to tokenized text
    hashing_tf = HashingTF(inputCol="words", outputCol="tf_features")
    hash_tf = hashing_tf.transform(words_data)
    idf = IDF(inputCol="tf_features", outputCol="tf_idf_features")
    idf_model = idf.fit(hash_tf)
    tf_idf = idf_model.transform(hash_tf)

    # preprocess test data
    test_data = test_data.filter(col("comment_text") != '"').withColumn('id', monotonically_increasing_id())
    test_words = tokenizer.transform(test_data).select("id", "words")
    test_hash_tf = hashing_tf.transform(test_words)
    test_tf_idf = idf_model.transform(test_hash_tf)
    test_res = test_data.select("id")

    # train logistic regression model and make predictions on test data
    reg_param = 0.1
    for the_column in out_cols:
        lr = LogisticRegression(featuresCol="tf_idf_features", labelCol=the_column, regParam=reg_param)
        lr_model = lr.fit(tf_idf.limit(4000))
        res_test = lr_model.transform(test_tf_idf)
        extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
        res_test = res_test.withColumn("proba" + '_' + the_column, extract_prob("probability")).withColumn("pred" + '_' + the_column, col('prediction'))
        res_test = res_test.select(res_test["id"], res_test["proba" + '_' + the_column], res_test["pred" + '_' + the_column])
        test_res = test_res.join(res_test, on=["id"])

    # display sample predictions
    test_res.sample(False, 0.4).show()


if __name__ == "__main__":
    main()