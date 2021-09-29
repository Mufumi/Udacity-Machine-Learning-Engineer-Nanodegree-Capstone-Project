# Udacity Machine Learning with Azure Capstone Project

<p align="center">
  <img width="200" src="https://github.com/Mufumi/Udacity-Optimizing-a-ML-Pipeline-in-Azure-Tutorial/blob/main/Azure%20Ml.jpg" alt="Azure Ml">
</p>

## Overview

This project is part of the Udacity Azure ML Nanodegree program. In this project, we deploy an Azure ML model using the Python SDK trained with a local training script. The model's objective is to predict the probability of a track being liked on my Spotify based on a playlist of liked and disliked tracks. This model is then compared to an Azure AutoML model.

### Dataset

The dataset is obtained from wrangling data from my Spotify playlist with the concept referenced from a project by [Brice Vergnou](https://github.com/Brice-Vergnou/spotify_recommendation)

The [Spotify Web API](https://developer.spotify.com/console/get-audio-features-several-tracks/?ids=) can be used to find track characteristics. The features of the dataset include:

* danceability
* energy
* key
* loudness
* mode
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo

To  generate this playlist, you must have a Spotify Account and use the Web API developer options. For the _Get Audio features_ you must authenticate your API request and obtain the OAuth token.

This [Notebook](https://github.com/Mufumi/Udacity-Capstone-Project/blob/e9d1df4f76dd3e546f114077ff47e8b3ebab0dc6/Spotify%20Dataset%20Wrangler.ipynb) shows how creating dataset from you personal playlist can be made.

### Data access into Azure

For importing the data, I used the `TabularDatasetFactory` class, importing data using the `from_delimited_files` method. The class contains methods to create a tabular dataset for Azure Machine Learning

## Architecture 
The architecture of the project can be seen here:
<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Architecture.png" alt="Capstone Project architecture">
</p>

## Hyperdrive Experiment

### Dependencies 

The [dependencies file](https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Capstone_Project_Files/starter_file/conda_dependencies.yml) lists all packages that need to be installed before the hyperdrive experiment can be run.

The data is obtained from a csv file and converted to structured data using a dataframe. Because the data is regularized and featurized, there was no need to clean the data. The data was then split into a training set and test set, with the Logistic Regression model chosen as the classifier. The hyperparameters for this model were the Regularization Strength and Max iterations parameters, assisting in the convergence of the model. The best performance model had a regularization strength of 0.5247 and max iterations of 1000 as illustrated.

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/HD_Run_Widget.png" alt="HD Run Details">
</p>


**Parameter sampler and stopping policy chosen**

The random parameter sampler was chosen based on its conservative usage to computing resources. The early stopping policy chosen was the Bandit policy 
to ensure that the experiments run within specific threshold. 

## AutoML Experiment

### AutoML settings

The AutoML experiment was set to timeout after 30 minutes was initiated using the automl compute instance, using compute instance using DS3_v2 machine, which is a genral purpose machine that is ideal for generic AutoML runs.

### AutoML configuration

The AutoML model task was set to `classification`, with the `primary_metric` set to accuracy using the auto_ml_ds `training_data`. This experiment's `label_column_name` was set to the **liked** column and had a default 8 `cross-validation` metric.

Using AutoML, the model of choice was the Voting Ensemble classifier which is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output. The best run had an ID of `AutoML_19b2c7bc-4f1a-47b9-8e97-63d12892f11f_39`

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/AutoML_Run_Widget.png" alt="AML Run Details">
</p>

The deployed model was the one computed in the AutoML experiment with endpoints set to _active_ as illustrated

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Azure-ML-Active-REST-endpoint.png" alt="Endpoint Active">
</p>

## Model comparison

Initially, training the model using a script and tuning the hyperparameters using HyperDrive, resulted in an accuracy of 0.85 after 35 minutes 11 seconds
The Auto ML model produced an accuracy of 0.845 within 29 seconds. After changing the `max_iter` to sweep between 1 and 100, the Hyperdrive experiment's best run had an accuracy of 0.925 within 58 minutes and 26 seconds, whilst the AutoML run had an accuracy of 0.92 within 24 minutes. The HyperDirve parameter optimizer performed with a higher accuracy but required more time and demanded a lot more iterations for tuning the hyperparameters.

## Screen-cast

[![Deploying a model using Azure Studio](deploy_model_using_python_sdk.png)](https://youtu.be/UqRCffzliro)

## Future work
* For the training model, the selected model was a Logistic Regression model which performed well considering its accuracy. Alternative classification algorithms can also be considered especially those that require less time to process.
* Consider using `AUC_Weighted` as primary metric as it does not discriminate on training data

* Unfortunately Spotify API has max request for 100 tracks meaning the preliminary dataset has 200 entries which is not enough data for a machine learning model with high predictive confidence.

## Proof of service clean up
Delete method was used in code but will only occur once the script run is completed
