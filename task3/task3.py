# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import sys
from operator import add

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
# reload(sys)
# sys.setdefaultencoding('utf8')
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Logistic Regression Adult Dataset <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("task3")\
        .getOrCreate()


    # access csv file in spark datafram

    train_df = spark.read.csv(sys.argv[1], header = False, inferSchema = True)

    column_names = ["age", "workClass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation", "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "Labels"]
    train_df = train_df.toDF(*column_names)

    test_df = spark.sparkContext.textFile(sys.argv[2]).map(lambda line : line.split(","))
    test_df = test_df.filter( lambda line : len(line) >2 )
    test_df = test_df.toDF()
    test_df = test_df.toDF(*column_names)

    res_df = train_df.union(test_df)

    def preProcess(spark_df):


        # strip spaces from all columns
        c_names = spark_df.columns
        for colname in c_names:
            spark_df = spark_df.withColumn(colname, F.trim(F.col(colname)))

        # drop irrelevant columns

        spark_df = spark_df.drop("fnlwgt","education")

        c_names.remove("fnlwgt")
        c_names.remove("education")

        # replace "?" with null

        for colname in c_names:
            spark_df = spark_df.withColumn(colname,F.when(F.col(colname) == "?", None).otherwise(F.col(colname)))

        # now drop all the null values

        spark1_df = spark_df.na.drop("any")

        # get categorical features
        cat_cols = ["workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"]

        # put indices for categories in a feature
        string_indexer = StringIndexer(inputCols = cat_cols, outputCols = ["wc_index", "mS_index", "occ_index", "rel_index", "race_index", "sex_index", "natCou_index"])
        spark2_df = string_indexer.fit(spark1_df).transform(spark1_df)

        # drop the original features
        for colname in cat_cols:
            spark2_df = spark2_df.drop(colname)

        # rename the columns for convinience
        new_columns = ["age", "educationNum","capitalGain", "capitalLoss", "hoursPerWeek", "Labels", "workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"]
        spark2_df = spark2_df.toDF(*new_columns)

        # get rid of sparse columns capitalGain and capitalLoss
        spark2_df = spark2_df.drop("capitalGain","capitalLoss")

        # get columns that need to be converted to float
        float_cols = spark2_df.columns

        # convert all except "Labels" to float values
        for colname in float_cols:
            if colname == "Labels":
                continue

            spark2_df = spark2_df.withColumn(colname, F.col(colname).cast("float"))


        # modify the "Labels" columns to suit the ML algorithms

        spark3_df = spark2_df.withColumn("Labels", F.when((F.col("Labels") == "<=50K") | (F.col("Labels") == "<=50K."), 0).otherwise(F.col("Labels"))).withColumn("Labels", F.when((F.col("Labels") == ">50K")| (F.col("Labels") == ">50K."), 1).otherwise(F.col("Labels")))
        # now change vlaues to integer types
        spark3_df  = spark3_df.withColumn("Labels",F.col("Labels").cast("int"))

        # now one-hot encoding of categorical features
        encoder = OneHotEncoder(inputCols = cat_cols, outputCols = ["wc_index", "mSindex", "occ_index", "rel_index", "race_index", "sex_index", "nC_index"])

        X_train = encoder.fit(spark3_df).transform(spark3_df)

        # drop original categorical columns
        for colname in cat_cols:
            X_train = X_train.drop(colname)

        # now all the columns except the "Labels" are to be used as training feature
        all_cols = X_train.columns
        train_cols = list()
        for colname in all_cols:
            if colname == "Labels":
                continue

            train_cols.append(colname)

        print(train_cols)

        vector_assembler = VectorAssembler(inputCols = train_cols, outputCol = "features")
        X1_train = vector_assembler.transform(X_train)



        return X1_train



    final_df = preProcess(res_df)
    test_train_split = final_df.randomSplit([0.7,0.3],47)
    X_train = test_train_split[0]
    X_test = test_train_split[1]

    lr = LogisticRegression(featuresCol = "features", labelCol = "Labels", regParam = 0.4, maxIter = 100)

    lrModel = lr.fit(X_train)

    coeffs = lrModel.coefficients
    intercept = lrModel.intercept

    print("Coefficients:",coeffs)
    print("intercept:",intercept)

    y_pred = lrModel.transform(X_test)
    y_pred.show()
    y_pred.printSchema()

    accuracy = y_pred.filter(F.col("Labels") == F.col("prediction")).count() / y_pred.count()

    
    print("**********")
    print()
    print("train-test split = (70-30)%")
    print("classifier : Logistic Regression")
    print()
    print("accuracy of classification, on test set: ",accuracy)
    print()
    print("**********")


    spark.stop()