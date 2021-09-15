# Udacity-Capstone-Project
Solving a problem of interest using Azure machine learning

Include overview

Creating track list classifier
Reference to Brice Vergnou project https://github.com/Brice-Vergnou/spotify_recommendation for idea



Spotify WEb API for finding track characteristics: https://developer.spotify.com/console/get-audio-features-several-tracks/?ids=

Must have a Spotify Account
Use developer properties
Create OAuth (authentication)
Unfortunately Spotify API has maz request for 100 tracks

Notebook for creating playlist :

<p align="center">
  <img width="200" src="https://github.com/Mufumi/Udacity-Optimizing-a-ML-Pipeline-in-Azure-Tutorial/blob/main/Azure%20Ml.jpg" alt="Azure Ml">
</p>

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Architecture 

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/Images/Screenshot from 2021-09-08 19-15-37.png" alt="Capstone Project architecture">
</p>

Images/Screenshot from 2021-09-08 19-15-37.png

## Summary
This dataset contains customer data that is going to be used to find the best strategies to improve for the next marketing campaign. The aim is to predict the effectiveness of the current marketing campaign.

The best performing model was a Logistic Regression model with the hyperparameters obtained from Hyperdrive.
## Scikit-learn Pipeline

The data is obtained from a csv file and converted to structured data using a dataframe. It was then cleaned and a OneHotEncoder technique from the sklearn library was implemented to label some features of the data. Once cleaned the data was split into a training set and test set, with the Logistic Regression model chosen as the classifier. The hyperparameters for this model were the Regularization Strength and Max iterations parameters, assisting in the convergence of the model. The best performance model had a regularization strength of 0.3711 and max iterations of 1000.

**Parameter sampler and stopping policy chosen**

The random parameter sampler was chosen based on its conservative usage to computing resources. The early stopping policy chosen was the Bandit policy 
to ensure that the experiments run within specific threshold. 

## AutoML

Using AutoML, the model of choice was the XGBoost classifier which is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. For data transformation, AutoML used the SKLearn Sparse Normalizer

## Pipeline comparison

Training the model using a script and tuning the hyperparameters using HyperDrive, resulted in an accuracy of 0.9180 after 1 minutes 44 seconds
The Auto ML model produced an accuracy of 0.91563 within 29 seconds. The HyperDirve parameter optimizer performed with a higher accuracy but required more time and demanded a lot more iterations for tuning the hyperparameters.


## Future work
For the training model, the selcted model was a Logistic Regression model which performed well considering its accuracy. Alternative classification algorithms can also be considered especially those that require less time to process. Additionally, the training data could have benefited from being encoded using the One Hot Encoder library instead of manually performing the cleaning.

## Proof of cluster clean up
Delete method was used in code but will only occur once the script run is completed

