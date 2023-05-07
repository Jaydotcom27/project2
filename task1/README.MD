# Part 1 : TOXIC COMMENT CLASSIFICATION

This folder contains the following files:
* *README file*: This file
* *Files*: task1.py & test1.sh

## Data Source
You can find the data from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## Required libraries
To run these codes, it is mandatory to run the following libraries:
* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.feature import LogisticRegression
* import pyspark.sql.functions as psf
* import sys

## Instructions to use this code
To run everything we call the test file as `bash test1.sh` and it will get the input, process the information through the Spark framework and output the final result. A copy of each dataset named “test.csv” and “train.csv” must be in this folder.