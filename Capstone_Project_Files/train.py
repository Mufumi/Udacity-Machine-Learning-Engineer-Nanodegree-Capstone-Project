from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

url_path='https://raw.githubusercontent.com/Mufumi/Udacity-Capstone-Project/main/Spotify_playlist/spotify_playlist.csv'

# Reading the downloaded content and turning it into a pandas dataframe

track_df = TabularDatasetFactory.from_delimited_files(path=url_path)

track_df = track_df.to_pandas_dataframe()

x=track_df.iloc[:,:-1]
y=track_df.iloc[:,-1:]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")
#I increased the number of iterations as regression model could not converge
    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
        
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
