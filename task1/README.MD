# Toxic Comment Classification (Part 1)

This repository contains one script and two csv files to classify the comments into the level of toxicity. One csv file is used for training and the other one for testing the machine learning logistic regression model.

## Getting Started

Proceed to clone this folder into the Cloud cluster spark-examples folder. In case you are running it locally, its neccesary to import the following libraries:

* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.feature import LogisticRegression
* import pyspark.sql.functions as psf
* import sys

## Data 

The data used in this analysis was obtained from a Kaggle Toxic Comment Classification Challenge, which can be accesed here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## Usage

To run this code you need a functional Spark system running, for our experiments a 3 node cluster hosted on Google Cloud was the way to go. Two worker nodes and one manager orchestrating.
The driver code is stored in the file `test1.sh` and it will load the input data, train the machine learning model through the Spark Framework and output the classification for the 'test.csv' file. Make sure to double check the appropiate location for the two datafiles csv, it needs to be into the task1 folder with the names 'test.csv' and 'train.csv'.



