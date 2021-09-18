# Udacity Machine Learning with Azure Capstone Project
Solving a problem of interest using Azure machine learning

<p align="center">
  <img width="200" src="https://github.com/Mufumi/Udacity-Optimizing-a-ML-Pipeline-in-Azure-Tutorial/blob/main/Azure%20Ml.jpg" alt="Azure Ml">
</p>

## Overview

This project is part of the Udacity Azure ML Nanodegree program. In this project, we deploy an Azure ML model using the Python SDK trained with a local training script. The model's objective is to predict the probability of a track being liked on my Spotify based on a playlist of liked and disliked tracks. This model is then compared to an Azure AutoML model.

### Dataset

The dataset is obtained from wrangling data from my Spotify playlist with the concept referenced from a project by [Brice Vergnou](https://github.com/Brice-Vergnou/spotify_recommendation)

The [Spotify Web API](https://developer.spotify.com/console/get-audio-features-several-tracks/?ids=) can be used to find track characteristics. The features of the dataset include:

*danceability
*energy
*key
*loudness
*mode
*speechiness
*acousticness
*instrumentalness
*liveness
*valence
*tempo

To  generate this playlist, you must have a Spotify Account and use the Web API developer options. For the _Get Audio features_ you must authenticate your API request and obtain the OAuth token.

Notebook for creating playlist can be found [here](https://github.com/Mufumi/Udacity-Capstone-Project/blob/e9d1df4f76dd3e546f114077ff47e8b3ebab0dc6/Spotify%20Dataset%20Wrangler.ipynb)

## Architecture 

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Architecture.png" alt="Capstone Project architecture">
</p>


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

## Screen-cast

[![Deploying a model using Azure Studio](deploy_model_using_python_sdk.png)](https://youtu.be/UqRCffzliro)

## Future work
For the training model, the selcted model was a Logistic Regression model which performed well considering its accuracy. Alternative classification algorithms can also be considered especially those that require less time to process. Unfortunately Spotify API has max request for 100 tracks meaning the preliminary dataset has 200 entries which is not enough data for a machine learning model with high predictive confidence.

## Proof of service clean up
Delete method was used in code but will only occur once the script run is completed
