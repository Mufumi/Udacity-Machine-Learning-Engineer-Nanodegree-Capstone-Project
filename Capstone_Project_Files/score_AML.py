
import joblib
import json
import numpy as np
import pandas as pd
import sklearn

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION)
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models)
    model_path = '/home/mufumi/Documents/Mufumi_2021/Udacity/Azure ML Capstone Project/Github_repo/Udacity-Capstone-Project-main/Capstone_Project_Files/starter_file/auto-ml-best_run.pkl'
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

    # Note here, the entire source directory from inference config gets added into image.
    # Below is an example of how you can use any extra files in image.
    
    with open('/home/mufumi/Documents/Mufumi_2021/Udacity/Azure ML Capstone Project/Github_repo/Udacity-Capstone-Project-main/Capstone_Project_Files/testdata.json') as json_file:
        #I used the absolute path of the file here. Change to specific file location
        data = json.load(json_file)

input_sample = {
    "danceability":0.724,
    "energy":0.6,
    "key":1,
    "loudness":-6.25,
    "mode":0,
    "speechiness":0.087,
    "acousticness":0.28,
    "instrumentalness":6.83e-05,
    "liveness":0.108,
    "valence":0.201,
    "tempo":164.037
}
output_sample = 0

@input_schema('data',PandasParameterType(pd.DataFrame([input_sample])))
@output_schema(StandardPythonParameterType(output_sample))

def run(data):
    try:
        result = int(model.predict(data))
        
        if result==0:
            prediction='Disliked'
        else:
            if result==1:
                prediction ='Liked'
            else:
                prediction ='Unclassified'
        result_string = "The track has features that you {}".format(prediction)
        return  result_string
    
    except Exception as e:
        error = str(e)
        return error