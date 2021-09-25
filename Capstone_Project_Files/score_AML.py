import joblib
import json
import numpy as np

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION)
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models)
    model_path = './auto-ml-best_run.pkl'#os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_run_hd.pkl')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

    # Note here, the entire source directory from inference config gets added into image.
    # Below is an example of how you can use any extra files in image.
    with open('./source_directory/testdata.json') as json_file:
        loaded_data = json.load(json_file)

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

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(loaded_data):
    try:
        loaded_data_df=pd.DataFrame([loaded_data])
        result = model.predict(loaded_data_df)
        prediction = 'Disliked'
        if result==0:
            prediction=prediction
        else:
            prediction='Liked'
        # You can return any JSON-serializable object.
        return "The track has features that you " + prediction
    except Exception as e:
        error = str(e)
        return error


