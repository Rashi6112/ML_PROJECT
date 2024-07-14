# Sonar-Rock-Vs-Mine-Prediction with Python

This repository contains a project aimed at classifying objects in sonar data as either rocks or mines using machine learning, specifically Logistic Regression.

## Table of Contents
1. [Introduction](#introduction)
2. [Importing the Libraries](#importing-the-libraries)
3. [Data Collection and Data Processing](#data-collection-and-data-processing)
   - 3.1 [Loading the Dataset](#loading-the-dataset)
   - 3.2 [Basic Statistics of the Data](#basic-statistics-of-the-data)
   - 3.3 [Checking for Null Values](#checking-for-null-values)
   - 3.4 [Counts of Unique Values in the '60' Column](#counts-of-unique-values-in-the-60-column)
4. [Splitting the Data](#splitting-the-data)
   - 4.1 [Train-Test Split](#train-test-split)
5. [Model Training: Logistic Regression](#model-training-logistic-regression)
6. [Model Evaluation](#model-evaluation)
   - 6.1 [Accuracy on Training Data](#accuracy-on-training-data)
   - 6.2 [Accuracy on Test Data](#accuracy-on-test-data)
7. [Making Predictions](#making-predictions)
   - 7.1 [Prediction for Input Data 1](#prediction-for-input-data-1)
   - 7.2 [Prediction for Input Data 2](#prediction-for-input-data-2)
8. [Project Aim](#project-aim)
9. [Installation](#installation)

## Introduction
This project aims to classify sonar returns from rocks and metal cylinders using the Logistic Regression algorithm. The data comprises sonar signals reflected from a metal cylinder or rock under various conditions and angles.

## Importing the Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
