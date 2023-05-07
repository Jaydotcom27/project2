#!/bin/bash

#********************


source ../../../env.sh
hdfs dfs -rm -r /task2/input/
hdfs dfs -rm -r /task2/output/

hdfs dfs -mkdir -p /task2/input/

hdfs dfs -put ./heart_dataset.csv /task2/input

/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./task2.py hdfs://$SPARK_MASTER:9000/task2/input/heart_dataset.csv