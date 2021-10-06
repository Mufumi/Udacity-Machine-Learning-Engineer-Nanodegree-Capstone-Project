# Udacity Machine Learning with Azure Capstone Project

<p align="center">
  <img width="200" src="https://github.com/Mufumi/Udacity-Optimizing-a-ML-Pipeline-in-Azure-Tutorial/blob/main/Azure%20Ml.jpg" alt="Azure Ml">
</p>

## Overview

This project is part of the Udacity Azure ML Nanodegree program. In this project, we deploy an Azure ML model using the Python SDK trained with a local training script. The model's objective is to predict the probability of a track being liked on my Spotify based on a playlist of liked and disliked tracks. This model is then compared to an Azure AutoML model.

## Project Set up and Installation

For this project, the Microsoft Azure Machine Learning tool must be accesible within Azure. Once istalled the [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/) can also be installed to assist the user create and manage Azure resources.

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

The [dependencies file](https://github.com/Mufumi/Udacity-Capstone-Project/blob/a82dd586b9e25c3a9aa54b8906f0f342d8a99532/Capstone_Project_Files/starter_file/conda_dependencies.yml) lists all packages that need to be installed before the hyperdrive experiment can be run.

The data is obtained from a csv file and converted to structured data using a dataframe. Because the data is regularized and featurized, there was no need to clean the data. The data was then split into a training set and test set, with the Logistic Regression model chosen as the classifier. The hyperparameters for this model were the Regularization Strength and Max iterations parameters, assisting in the convergence of the model. The best performance model had a regularization strength of 0.7542 and max iterations of 197 as illustrated.

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/HD_Best_Run.png" alt="HD Best Run">
</p>

### Run Details

The RunDetails widget was used to show the progress of the tuning as illustrated:

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/HD_Run_Widget.png" alt="HD Run Details Widget">
</p>

**Parameter sampler and stopping policy chosen**

The random parameter sampler was chosen based on its conservative usage to computing resources. The early stopping policy chosen was the Bandit policy 
to ensure that the experiments run within specific threshold. 


**Bandit Policy arguments**
`slack_factor` is the ratio used to calculate the allowed distance from the best performing experiment run and the `evaluation_interval` is the frequency for applying the policy. For this experiment, the slack allowed for preceeding models was selected to be within 10% of the best model. The interval of the policy was set to reevaluating after 2 models. Both these choices were selected with resource optimization as a primary goal of the experiment.

**Random parameter sampling arguments**

_Regularization Strength(--C)_

Regularization is a form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero. In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. For this experiment, the coefficients were scaled to be between 0 and 1, to simplify an understanding of the model.

_Maximum iterations(--max_iter)_

This is the maximum number of iterations taken for the solvers to converge. Every chosen `max_iter` value is a random value between 1 and 1000. A higher value is expected to produce a higher primary scoring metric score. We do, however, have to find a conservative estimate of this value, explaining why we're iterating random values over this range.

### HyperDrive Settings

Withe the parameter smapling and early termination policy set, we can configure a HyperDrive Run. This run's primary goal is to _maximize_ the primary metric is `Accuracy`. The configuration was also set to iterate of a maximum of 200 runs. Based on the constraints and application of the experiment, the user can choose the `max_runs` and `primary_metric_name` variables.

## AutoML Experiment

### AutoML settings

The AutoML experiment was set to timeout after 30 minutes was initiated using the automl compute instance, using compute instance using DS3_v2 machine, which is a genral purpose machine that is ideal for generic AutoML runs.

### AutoML configuration

The AutoML model task was set to `classification`, with the `primary_metric` set to accuracy using the auto_ml_ds `training_data`. This experiment's `label_column_name` was set to the **liked** column and had a default 8 `cross-validation` metric.

Using AutoML, the model of choice was the Voting Ensemble classifier which is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output. The best run had an ID illustrated below:

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/AutoML_Run_Widget.png" alt="AML Run Details">
</p>

The best run and best model can be illustrated as:

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Auto_ML_best_run_best_model.png" alt="AML Run Details">
</p>


The deployed model was the one computed in the AutoML experiment with endpoints set to _active_ as illustrated

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Azure-ML-Active-REST-endpoint.png" alt="Endpoint Active">
</p>

## Model Deployment

Once the best model is obtained, we can proceed to register it in the workspace. For model deployment, we have to set up a environment that will host the model. A constant variable `CONDA_ENV_FILE_PATH`, has the path to packages invloved in the model build. We can download this file and use it to create an environment with the recquired packages and dependecies. Azure also generates a scoring file based on the inputs of the model. We can use this file as an input variable to the inference config.

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Deploying_Model.png" alt="Deploying Model">
</p>

### Inference config

The `inference_config` object represents configuration settings for a custom environment used for deployment. It translates the behaviour of the model to a service which has interaction capability. The scoring file generated by the AutoML is available for download within the experiment. The configuration also requires the environment dependencies for deploying the model.

### ACI web service

For this exercise, I deployed the model as a Azure Container Instance (ACI) service. The service represents a machine learning model deployed as a web service endpoint on Azure Container Instances.

A deployed service is created from a model, script, and associated files. The resulting web service is a load-balanced, HTTP endpoint with a REST API. You can send data to this API and receive the prediction returned by the model. Considering the simplicity of the model, only one CPU core with 1 GB RAM was allocated.

**However, you did not include instructions on how to query the endpoint - the steps taken to query the endpoint, including how to compose the request, with sample data, sample response and interpretation of the response. Remember that providing code snippets and screenshots are every important here. You could also talk about the inference configuration.**


Once created, the user can get the service details:

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Service_Details.png" alt="Service Details">
</p>

Once the model is deployed as a service, a sample input test is conducted to ensure that the web service is operational. Data with json format is dumped into the model. With the correct **uri key**, **input_data** and **header** format, the data can be sent to the service API to intiate a **post request** from the service. The expected output is either a **0** (for prediction that track will be disliked) or a **1** (for prediction that track will be liked)

<p align="center">
  <img width="600" src="https://github.com/Mufumi/Udacity-Capstone-Project/blob/main/Images/Querying_the_endpoint.png" alt="Endpoint Query">
</p>

## Model comparison

Initially, training the model using a script and tuning the hyperparameters using HyperDrive, resulted in an accuracy of 0.85 after 35 minutes 11 seconds
The Auto ML model produced an accuracy of 0.845 within 29 seconds. After changing the `max_iter` to sweep between 1 and 100, the Hyperdrive experiment's best run had an accuracy of 0.925 within 58 minutes and 26 seconds, whilst the AutoML run had an accuracy of 0.92 within 24 minutes. The HyperDirve parameter optimizer performed with a higher accuracy but required more time and demanded a lot more iterations for tuning the hyperparameters.

## Screen-cast

[![Deploying a model using Azure Studio](deploy_model_using_python_sdk.png)](https://youtu.be/480CUqkTDZU)

## Future work
* For the training model, the selected model was a Logistic Regression model which performed well considering its accuracy. Alternative classification algorithms can also be considered especially those that require less time to process.
* Consider using `AUC_Weighted` as primary metric as it does not discriminate on training data

* Unfortunately Spotify API has max request for 100 tracks meaning the preliminary dataset has 200 entries which is not enough data for a machine learning model with high predictive confidence.

## Proof of service clean up
Delete method was used in code but will only occur once the script run is completed

## Project Wiki

The [project wiki](https://github.com/Mufumi/Udacity-Capstone-Project.wiki.git) has extra details of how I went about with the project.
