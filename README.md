# explainable-AI
This repository contains work on Explainable Artificial Intelligence on both Machine Learning and Deep Learning models

## 1. Extract Feature Relevance Extraction through Explainability of ML model
* **File**: _Feature_Relevance_Extraction_through_Explainability.ipynb_

This implementation takes a publicly available financial (bank) dataset from the UC Irvine Machine Learning Repository ([link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip)) and builds a machine learning classification model on this data. Using explainable AI (_Shapley Additive Values_ method), the _relevance_ of each of the features in the feature set are determined to understand, which features are important for a particular outcome. The details of the implementation are as follows:

### Dataset
* y-value: If the client has subscribed for a _term deposit_
* x-value: Set of _20_ features like age, marital status, education _etc._

### Classification steps
* The data is plotted and understood first to detect imbalance, missing values etc.
* The data is then Pre-processed/Cleaned
* Missing values are added to the data using _SMOTE_ method
* Using _logistic regression_ a classification model is built on this dataset
* A _grid search_ strategy is used to find the optimal hyper-parameters for the logistic regression model

### Explainability and feature relevance
* _SHAP_ (Shapley Additive Values) algorithm is used for explainability
* Each prediction made by the logistic regression model is explained in terms of the relevance/ importance of each feature in the dataset
