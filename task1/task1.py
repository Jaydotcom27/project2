from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col, monotonically_increasing_id

import pyspark.sql.functions as F
import pyspark.sql.types as T
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("task1")\
        .getOrCreate()

train = spark.read.csv(sys.argv[1], header = True)
test = spark.read.csv(sys.argv[2], header = True)

out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]
tok = Tokenizer(inputCol = "comment_text", outputCol = "words")
word = tok.transform(train)
train = train.withColumn("toxic", col("toxic").cast("int")).withColumn("severe_toxic", col("severe_toxic").cast("int")).withColumn("obscene", col("obscene").cast("int")).withColumn("threat", col("threat").cast("int")).withColumn("insult", col("insult").cast("int")).withColumn("identity_hate", col("identity_hate").cast("int"))
train = train.na.drop("any")

wordsData =  tok.transform(train)
hashingTF = HashingTF(inputCol = "words", outputCol = "R")
hash_Tf = hashingTF.transform(wordsData)
idf = IDF(inputCol = "R", outputCol = "features")

idfModel = idf.fit(hash_Tf)
tFidf = idfModel.transform(hash_Tf)

test = test.filter(col("comment_text")!= '"').withColumn('UID', monotonically_increasing_id())
test_tokens = tok.transform(test)
test_tf = hashingTF.transform(test_tokens)
test_tFidf = idfModel.transform(test_tf)
test_res = test.select("UID")

REG = 0.1
for the_column in out_cols:
    lr = LogisticRegression(featuresCol = "features", labelCol= the_column, regParam = REG)
    lrModel = lr.fit(tFidf.limit(4000))
    res_test = lrModel.transform(test_tFidf)
    extract_prob = F.udf( lambda x : float(x[1]), T.FloatType())

    res_test = res_test.withColumn("proba"+'_'+the_column,extract_prob("probability")).withColumn("pred"+'_'+the_column,col('prediction'))
    res_test = res_test.select(res_test["UID"], res_test["proba"+'_'+the_column], res_test["pred"+'_'+the_column])
    test_res = test_res.join(res_test, on=["UID"])

test_res.sample(False,0.4).show()