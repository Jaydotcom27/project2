from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler

import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("task2")\
        .getOrCreate()

heart_df = spark.read.csv(sys.argv[1],header = True)

heart_df = heart_df.na.drop("any")
a = heart_df.columns
b = a[:-1]
c = a[-1]

for item in b:
    heart_df = heart_df.withColumn(item,col(item).cast("float"))

heart_df = heart_df.withColumn("TenYearCHD", col("TenYearCHD").cast("int"))

vector_assembler = VectorAssembler(inputCols = b, outputCol = "SS_features")
temp_heart_df = vector_assembler.transform(heart_df)

standard_scaler = StandardScaler(inputCol = "SS_features", outputCol = "scaled")
temp_heart_df_scaled = standard_scaler.fit(temp_heart_df).transform(temp_heart_df)

lr = LogisticRegression(featuresCol = "scaled", labelCol = "TenYearCHD", regParam = 0.2, maxIter = 10)
lrModel = lr.fit(temp_heart_df_scaled)
AVGpredictions = lrModel.transform(temp_heart_df_scaled).select(mean('prediction')).withColumnRenamed('avg(prediction)','Overall risk')
ModelWeights = list(map(lambda x: abs(x), lrModel.coefficients))
MostRelevantFeatures = sorted(list(zip(b, ModelWeights)), key= lambda x: x[1], reverse=True)

AVGpredictions.show()

print('\nMost Relevant Features for our Linear Reg. Model')
[print(column,'\t', feature_weight) for column, feature_weight in MostRelevantFeatures]