# Part 2 : HEART DISEASE PREDICTION USING LOGISTIC REGRESSION

This folder contains the following files:
* *README file*: This file
* *Files*: task2.py & test2.sh

## Data Source
You can find the data from:
 
https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data?resource=download&select=CHD_preprocessed.csv

## Required libraries
To run these codes, it is mandatory to run the following libraries:
* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.classification import LogisticRegression
* from pyspark.ml.feature import VectorAssembler
* import pyspark.sql.functions as psf
* import sys

## Instructions to use this code
To run everything we call the test file as `bash test2.sh` and it will get the input, process the information through the Spark framework and output the final result. A copy of each dataset named “heart_dataset.csv” must be in this folder.