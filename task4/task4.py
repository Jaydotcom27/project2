from __future__ import print_function
import sys
from operator import add



from pyspark.sql import SparkSession
import pyspark.sql.functions as sql_functions
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.mllib.stat import Statistics


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Logistic Regression Adult Dataset <file>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("task3")\
        .getOrCreate()
    
    train_df = spark.read.csv(sys.argv[1], header = False, inferSchema = True)

    column_names = ["age", "workClass", "fnlwgt", "education", "educationNum", "maritalStatus", "occupation", "relationship", "race", "sex", "capitalGain", "capitalLoss", "hoursPerWeek", "nativeCountry", "Income"]
    train_df = train_df.toDF(*column_names)

    test_df = spark.sparkContext.textFile(sys.argv[2]).map(lambda line : line.split(","))
    test_df = test_df.filter( lambda line : len(line) >2 )
    test_df = test_df.toDF()
    test_df = test_df.toDF(*column_names)

    df_runing = train_df.union(test_df)


    def preprocessingdata(spark_df):

        column_names_new=spark_df.columns
        for i in column_names_new:
            spark_df=spark_df.withColumn(i,sql_functions.trim(sql_functions.col(i)))
        
        #The dataset has a couple of values with value ?, replace with null values if applicable        
        for i in column_names_new:
            spark_df=spark_df.withColumn(i,sql_functions.when(sql_functions.col(i)=="?", None).otherwise(sql_functions.col(i)))

        #drop null values
        spark_df1a=spark_df.na.drop("any")    

        #Change Income class variable to 0 and 1
        spark_df1=spark_df1a.withColumn("Income", sql_functions.when((sql_functions.col("Income") == "<=50K") | (sql_functions.col("Income") == "<=50K."), 0).otherwise(sql_functions.col("Income"))).withColumn("Income", sql_functions.when((sql_functions.col("Income") == ">50K")| (sql_functions.col("Income") == ">50K."), 1).otherwise(sql_functions.col("Income")))


        #once I transform the categorical data to numeric data, transform capital gain and loss to log and scale all the numeric feaures. then perform titus technique to stay with most important features
        #already have the same feature in numeric
        spark_df1=spark_df1.drop("education")

        #Tranform categorical string data to numeric
        categorical_features=["workClass","maritalStatus","occupation", "relationship","race", "sex","nativeCountry"]
        categorical_index=StringIndexer(inputCols=categorical_features, 
                                        outputCols= ["workClass_num","maritalStatus_num","occupation_num", "relationship_num","race_num", "sex_num","nativeCountry_num"])
        spark_df2=categorical_index.fit(spark_df1).transform(spark_df1)

        for i in categorical_features:
            spark_df2=spark_df2.drop(i)

        new_columns =["age", "fnlwgt", "educationNum","capitalGain", "capitalLoss", "hoursPerWeek", "Income", 
                      "workClass", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"]
        spark_df2=spark_df2.toDF(*new_columns)

        #Log transform capital gain and capital loss adn fnlwgt.
        skewed = ['capitalGain', 'capitalLoss','fnlwgt']
        #scale the remaining numeric features
        spark_df2 = spark_df2.withColumn(skewed[0], sql_functions.log1p(sql_functions.col(skewed[0]))) \
                                      .withColumn(skewed[1], sql_functions.log1p(sql_functions.col(skewed[1]))) \
                                      .withColumn(skewed[2], sql_functions.log1p(sql_functions.col(skewed[2])))
        
        for i in spark_df2.columns:
            if i!="Income":
                spark_df2=spark_df2.withColumn(i, sql_functions.col(i).cast("float"))

        spark_df3=spark_df2.withColumn("Income",sql_functions.col("Income").cast("int"))

        #Link the new categorical features to numerical values
        encoder=OneHotEncoder(inputCols=categorical_features, 
                              outputCols=["workClass_num","maritalStatus_num","occupation_num", "relationship_num","race_num", "sex_num","nativeCountry_num"])

        #Incorporate new features to train dataset
        X_train=encoder.fit(spark_df3).transform(spark_df3)
        for i in categorical_features:
            X_train=X_train.drop(i)
        

        #Perform Pearson correlation test to stay with most important features
        col_names = X_train.columns

        # Initialize lists to store the results
        param = []
        correlation = []
        abs_cor = []
        for c in col_names:
            if c != "income":
                corr = abs(Statistics.corr(X_train.select('Income', c).rdd.map(lambda x: (float(x[0]), float(x[1]))), method="pearson"))
                param.append(c)
                correlation.append(corr[0][1])
                abs_cor.append(abs(corr[0][1]))

                
        # Create a DataFrame from the results
        param_df = X_train.createDataFrame(zip(correlation, param, abs_cor), ["correlation", "parameter", "abs_cor"])

        # Sort the DataFrame by absolute correlation in descending order
        param_df = param_df.orderBy(sql_functions("abs_cor").desc())

        # Set the 'parameter' column as the index
        param_df = param_df.withColumnRenamed("parameter", "index").drop("abs_cor").drop("correlation")

        best_features = param_df.select("index").limit(8).rdd.flatMap(lambda x: x).collect()


        #top 8 best features
        print("Top features to include in model:",best_features) 

        #Remove the class variable
        columns=best_features.columns
        train_columns=list()
        for i in columns:
            if i!="Income":
                train_columns.append(i)

        # Create a VectorAssembler to combine the features into a single vector column
        assembler = VectorAssembler(inputCols=train_columns, outputCol="features")
        final_X_train = assembler.transform(X_train)

        return final_X_train
    
    final_df = preprocessingdata(df_runing)
    test_train_split = final_df.randomSplit([0.8,0.2],47)
    X_train = test_train_split[0]
    X_test = test_train_split[1]

    decision_tree = DecisionTreeClassifier(featuresCol = "features", labelCol = "Labels")

    decision_tree_Model = decision_tree.fit(X_train)

    y_pred = decision_tree_Model.transform(X_test)
    y_pred.show()
    y_pred.printSchema()

    accuracy = y_pred.filter(sql_functions.col("Labels") == sql_functions.col("prediction")).count() / y_pred.count()

    print("**********")
    print()
    print("Classifier : Decision Tree")
    print()
    print("Accuracy of Decision Tree Model: ",accuracy)
    print()
    print("**********")

    RandomForest = RandomForestClassifier(featuresCol = "features", labelCol = "Labels", numTrees = 15)

    RandomForest_Model = RandomForest.fit(X_train)


    y_pred = RandomForest_Model.transform(X_test)
    y_pred.show()
    y_pred.printSchema()

    accuracy = y_pred.filter(F.col("Labels") == sql_functions.col("prediction")).count() / y_pred.count()

    print("**********")
    print()
    print("Classifier : Random Forest")
    print()
    print("Accuracy of Random Forest Model: ",accuracy)
    print()
    print("**********")

    spark.stop()


