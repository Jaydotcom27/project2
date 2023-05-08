# Census Income Dataset (Random Forest and Decision Tree) 

This repository contains one script and two csv datfiles to classify based on a set of feautures if the expected income is higher or lower than USD$50,000. Using two machine learning models: Random Forest and Decision Tree.

## Getting Started

Proceed to clone this folder into the Cloud cluster spark-examples folder. If you are running it locally, its required to import this libraries:

* from __future__ import print_function
* from pyspark.sql import SparkSession
* from pyspark.ml.classification import LogisticRegression
* from pyspark.ml.feature import StringIndexer, VectorAssembler
* import pyspark.sql.functions as psf
* import sys

## Data 

The data used in this analysis was obtained from the machine learning repository of the University of California, which can be accesed here:
https://archive.ics.uci.edu/ml/datasets/census+income

## Usage

To run this code you need a functional Spark system running, for our experiments a 3 node cluster hosted on Google Cloud was the way to go. Two worker nodes and one manager orchestrating.
The driver code is stored in the file `test4.sh`  and it will load the input datasets: 'adult.csv' and “adult_test.csv”, make sure to double check the appropiate location and name for the csv datafile, they need to be into the task4 folder.
