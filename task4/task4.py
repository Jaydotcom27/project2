from __future__ import print_function
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_functions
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier

def main(spark, input_path): 
    train_df = spark.read.csv(input_path, header = False, inferSchema = True)
    column_names = ["age", "workClass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation", "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "Labels"]
    train_df = train_df.toDF(*column_names)
    test_df = spark.sparkContext.textFile(sys.argv[2]).map(lambda line : line.split(","))
    test_df = test_df.filter( lambda line : len(line) >2 )
    test_df = test_df.toDF()
    test_df = test_df.toDF(*column_names)
    res_df = train_df.union(test_df)
    final_df = preProcess(res_df)

    # Random 80/20 data split
    test_train_split = final_df.randomSplit([0.8,0.2],47) # <- using same seed
    X_train = test_train_split[0]
    X_test = test_train_split[1]

    decision_tree = DecisionTreeClassifier(featuresCol = "features", labelCol = "Labels")

    decision_tree_Model = decision_tree.fit(X_train)

    y_pred = decision_tree_Model.transform(X_test)
    y_pred.show()
    y_pred.printSchema()

    accuracy = y_pred.filter(sql_functions.col("Labels") == sql_functions.col("prediction")).count() / y_pred.count()

    print("-----------------------------------------------------")
    print("Accuracy(DT): ",accuracy)
    print("-----------------------------------------------------")

    RandomForest = RandomForestClassifier(featuresCol = "features", labelCol = "Labels", numTrees = 15)

    RandomForest_Model = RandomForest.fit(X_train)


    y_pred = RandomForest_Model.transform(X_test)
    y_pred.show()
    y_pred.printSchema()

    accuracy = y_pred.filter(sql_functions.col("Labels") == sql_functions.col("prediction")).count() / y_pred.count()

    print("-----------------------------------------------------")
    print("Accuracy(RF): ",accuracy)
    print("-----------------------------------------------------")

    spark.stop()

def preProcess(spark_df):
    # Trim all the columns and drop not used
    c_names = spark_df.columns
    for colname in c_names:
        spark_df = spark_df.withColumn(colname, sql_functions.trim(sql_functions.col(colname)))
    spark_df = spark_df.drop("fnlwgt", "education")

    # Replace all the "?" with None and drop all rows with missing values
    for colname in spark_df.columns:
        spark_df = spark_df.withColumn(colname, sql_functions.when(sql_functions.col(colname) == "?", None).otherwise(sql_functions.col(colname)))
    spark_df = spark_df.na.drop("any")
    cat_cols = ["workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"]
    
    # Index all the categorical features and drop them
    string_indexer = StringIndexer(inputCols=cat_cols, outputCols=[f"{col}_index" for col in cat_cols])
    spark_df = string_indexer.fit(spark_df).transform(spark_df)
    spark_df = spark_df.drop(*cat_cols)

    # Rename the columns for convenience
    spark_df = spark_df.toDF("age", "educationNum","capitalGain", "capitalLoss", "hoursPerWeek", "Labels", "workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry")
    spark_df = spark_df.drop("capitalGain", "capitalLoss")

    # Convert all float columns to float
    for colname in spark_df.columns:
        if colname in ["Labels", "workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"]:
            continue
        spark_df = spark_df.withColumn(colname, sql_functions.col(colname).cast("float"))

    # Modify the "Labels" column to suit the ML algorithms
    spark_df = spark_df.withColumn("Labels", sql_functions.when(sql_functions.col("Labels").isin(["<=50K", "<=50K."]), 0).otherwise(sql_functions.col("Labels")))
    spark_df = spark_df.withColumn("Labels", sql_functions.when(sql_functions.col("Labels").isin([">50K", ">50K."]), 1).otherwise(sql_functions.col("Labels")))
    spark_df = spark_df.withColumn("Labels", sql_functions.col("Labels").cast("int"))

    # One-hot encode all the categorical features
    encoder = OneHotEncoder(inputCols=["workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"], 
                            outputCols=["wc_vec", "ms_vec", "occ_vec", "rel_vec", "race_vec", "sex_vec", "natCou_vec"])
    spark_df = encoder.fit(spark_df).transform(spark_df)
    spark_df = spark_df.drop("workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry")

    # All columns except "Labels" are used as training features
    assembler = VectorAssembler(inputCols=[col for col in spark_df.columns if col != "Labels"],
                                outputCol="features")
    spark_df = assembler.transform(spark_df)

    return spark_df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("RF and DT Adult Dataset", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession.builder.appName("RF and DT on Census Income Data").enableHiveSupport().getOrCreate()
    input_path = sys.argv[1]
    main(spark, input_path)


    