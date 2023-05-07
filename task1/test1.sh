#!/bin/bash

#********************


source ../../../env.sh
hdfs dfs -rm -r /task1/input/
hdfs dfs -rm -r /task1/output/

hdfs dfs -mkdir -p /task1/input/

hdfs dfs -put ./train.csv /task1/input
hdfs dfs -put ./test.csv /task1/input

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./task1.py hdfs://$SPARK_MASTER:9000/task1/input/train.csv hdfs://$SPARK_MASTER:9000/task1/input/test.csv