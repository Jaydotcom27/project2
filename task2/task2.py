from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
import sys

def main(spark, input_path):
    heart_df = spark.read.csv(input_path, header=True)
    heart_df = heart_df.na.drop("any")
    
    b = heart_df.columns[:-1]  
    for item in b:
        heart_df = heart_df.withColumn(item, col(item).cast("float"))

    # casting TenYearCHD for regression model
    heart_df = heart_df.withColumn("TenYearCHD", col("TenYearCHD").cast("int"))

    # create vector of input features
    vector_assembler = VectorAssembler(inputCols=b, outputCol="features")
    temp_heart_df = vector_assembler.transform(heart_df)

    # standardizing process
    standard_scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    temp_heart_df_scaled = standard_scaler.fit(temp_heart_df).transform(temp_heart_df)

    # train logistic regression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="TenYearCHD", regParam=0.2, maxIter=10)
    lr_model = lr.fit(temp_heart_df_scaled)

    # calculate the average risk prediction of the model
    avg_predictions = lr_model.transform(temp_heart_df_scaled).select(mean('prediction')).withColumnRenamed('avg(prediction)','Overall risk')

    # get the most relevant features for the model
    model_weights = list(map(lambda x: abs(x), lr_model.coefficients))
    most_relevant_features = sorted(list(zip(b, model_weights)), key= lambda x: x[1], reverse=True)

    # printing results
    avg_predictions.show()
    print('\nMost Relevant Features')
    [print(column,'\t', feature_weight) for column, feature_weight in most_relevant_features]

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Heart Disease Prediction using Logistic Regression").enableHiveSupport().getOrCreate()
    input_path = sys.argv[1]
    main(spark, input_path)