# Heart Disease Prediction Using Logistic Regression (Part 2)

This repository contains one script and one csv file to determine the principal set of features that are associated with a higher risk of developing coronary heart disease.

## Getting Started

Proceed to clone this folder into the Cloud cluster spark-examples folder. If you are running it locally, its required to import this libraries:

* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.classification import LogisticRegression
* from pyspark.ml.feature import VectorAssembler
* import pyspark.sql.functions as psf
* import sys

## Data 

The data used in this analysis was obtained from a preprocessed and cleaned data of Kaggle Logistic regression To predict heart disease, which can be accesed here:
https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data?resource=download&select=CHD_preprocessed.csv

## Usage

To run this code you need a functional Spark system running, for our experiments a 3 node cluster hosted on Google Cloud was the way to go. Two worker nodes and one manager orchestrating.
The driver code is stored in the file `test2.sh`  and it will load the input data, 'heart_dataset.csv', makee sure to double check the appropiate location and name for the csv datafile, it needs to be into the task2 folder.
