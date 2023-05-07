#!/bin/bash

#********************


source ../../../env.sh
hdfs dfs -rm -r /task3/input/
hdfs dfs -rm -r /task3/output/

hdfs dfs -mkdir -p /task3/input/

hdfs dfs -put ./adult.csv /task3/input/
hdfs dfs -put ./adult_test.csv /task3/input/

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./task3.py hdfs://$SPARK_MASTER:9000/task3/input/adult.csv hdfs://$SPARK_MASTER:9000/task3/input/adult_test.csv


hdfs dfs -rm -r /task3