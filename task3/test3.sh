#!/bin/bash
source ../../../env.sh
hdfs dfs -rm -r /task3/input/
hdfs dfs -rm -r /task3/output/
hdfs dfs -mkdir -p /task3/input/
hdfs dfs -put ./adult.csv /task3/input/
hdfs dfs -put ./adult_test.csv /task3/input/
/usr/local/spark/bin/spark-submit --master=spark://10.128.0.5:7077 ./task3.py hdfs://10.128.0.5:9000/task3/input/adult.csv hdfs://10.128.0.5:9000/task3/input/adult_test.csv
hdfs dfs -rm -r /task3