# Part 4 : Census Income Dataset

This folder contains the following files:
* *README file*: This file
* *Files*: task4.py & test4.sh

## Data Source
You can find the data from: https://archive.ics.uci.edu/ml/datasets/census+income

## Required libraries
To run these codes, it is mandatory to run the following libraries:
* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.feature import StringIndexer, VectorAssembler
* from pyspark.ml.classification import DecisionTreeClassifier,      RandomForestClassifier
* import pyspark.sql.functions as psf
* import sys

## Instructions to use this code
To run everything we call the test file as `bash test4.sh` and it will get the input, process the information through the Spark framework and output the final result. A copy of each dataset named “adult.csv” and “adult_test.csv” must be in this folder.