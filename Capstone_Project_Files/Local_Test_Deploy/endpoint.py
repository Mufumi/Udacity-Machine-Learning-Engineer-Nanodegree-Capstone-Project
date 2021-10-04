import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = "http://270b94e3-1631-4f04-9cff-90ef2e0a2d1e.southcentralus.azurecontainer.io/score"


# If the service is authenticated, set the key or token
key = ""

# Two sets of data to score, so we get two results back
data = {
    "data": [
    {
    "danceability":0.273,
    "energy":0.163,
    "key":7,
    "loudness":-15.889,
    "mode":1,
    "speechiness":0.0306,
    "acousticness":0.853,
    "instrumentalness":1.01e-06,
    "liveness":0.0835,
    "valence":0.202,
    "tempo":68.994
    }
  ]
}


# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {"Content-Type": "application/json"}
# If authentication is enabled, set the authorization header
headers["Authorization"] = f"Bearer {key}"

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
