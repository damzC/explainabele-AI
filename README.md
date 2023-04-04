# explainable-AI
This repository contains work on Explainable Artificial Intelligence on both Machine Learning and Deep Learning models

## 1. Extract Feature Relevance Extraction through Explainability of ML model
* **File**: Feature_Relevance_Extraction_through_Explainability.ipynb

This implementation takes a publicly available financial (bank) dataset from the UC Irvine Machine Learning Repository [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip) and builds a machine learning classification model on this data. Using explainable AI (Shapley Additive Values method), the relevance of each of the features in the feature set are determined to understand, which features are important for a particular outcome. The details of the implementation are as follows:

### Dataset
* y-value: If the client has subscribed for a term deposit
* x-value: Set of 20 features like age, marital status, education etc.

### Classification steps
* The data is plotted and understood first to detect imbalance, missing values etc.
* The data is then Pre-processed/Cleaned
* Missing values are added to the data using SMOTE method
* Using logistic regression a classification model is built on this dataset
* A grid search strategy is used to find the optimal hyper-parameters for the logistic regression model

### Explainability and feature relevance
* SHAP (Shapley Additive Values) algorithm is used for explainability
* Each prediction made by the logistic regression model is explained in terms of the relevance/ importance of each feature in the dataset
