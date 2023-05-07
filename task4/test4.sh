#!/bin/bash

#********************


source ../../../env.sh
hdfs dfs -rm -r /task4/input/
hdfs dfs -rm -r /task4/output/

hdfs dfs -mkdir -p /task4/input/

hdfs dfs -put ./adult.csv /task4/input/
hdfs dfs -put ./adult_test.csv /task4/input/

/usr/local/spark/bin/spark-submit --master=spark://10.128.0.5:7077 ./task4.py hdfs://10.128.0.5:9000/task4/input/adult.csv hdfs://10.128.0.5:9000/task4/input/adult_test.csv


hdfs dfs -rm -r /task4