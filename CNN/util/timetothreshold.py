import argparse
import matplotlib as mlp
import numpy as np
import pandas as pd
from scipy.integrate import simps #alternative to trapezoid rule
import math

def main(target, models, max_episodes, smooth, threshhold):
    """
    Main entry point for the application
    """
    #get target data and get relevant stats
    print(f"Getting max score for {target}")
    target_data = get_data_from_model_csv(target)
    if max_episodes != -1:
        target_data = np.resize(target_data, max_episodes)
    target_data = smooth_data(target_data, smooth)
    target_data_threshold = target_data[:int(target_data.shape[0]*threshhold)]
    target_max = np.max(target_data_threshold)
    target_max_achieved = find_first_occurence(target_max, target_data_threshold)

    #now we want to get all of the data from the model files
    model_ttt = {}
    for file in models:
        data = get_data_from_model_csv(file)
        if max_episodes != -1:
            data = np.resize(data, max_episodes)
        data = smooth_data(data, smooth)
        model_ttt[file] = find_first_occurence(target_max, data)

    #fin
    print(f"Target achieved a maximum score of {target_max} at episode {target_max_achieved}")
    for model in model_ttt:
        #didn't reach
        if model_ttt[model] == -1:
            continue
        print(f"{model} achieved {target_max} at episode {model_ttt[model]}")

def find_first_occurence(target, data):
    """
    Finds the first occurence of the target value in the data
    """
    for i in range(len(data)):
        if target == data[i]:
            return i
    return -1

def smooth_data(data, smooth):#
    """
    Sooth the data using the rolling window algorithm
    """
    data = pd.Series(data)
    smoothed_data = data.rolling(smooth, min_periods=1).mean()
    #smoothed_data = smoothed_data.iloc[smooth:] #Remove x elements used to smooth the data as they are NaN
    return np.array(smoothed_data)

def get_data_from_model_csv(path):
    """
    get and return a numpy array representing the data of the model.
    :param path: Path to the relevant CSV file
    """
    try:
        data = []
        with open(path, "r") as file:
            for row in file:
                data.append(float(row.split(",")[1]))
        return np.array(data)
    except FileNotFoundError:
        print(f"Error reading CSV file {path}. Quitting.")

if __name__ =="__main__":
    import os
    os.system("clear")
    parser = argparse.ArgumentParser(description="Calculate regret of a model")
    parser.add_argument("--target", metavar="target", required=True)
    parser.add_argument("--models", metavar="models",nargs="+" ,required=True)
    parser.add_argument("--episodes",default="-1", metavar="episodes",required=False)
    parser.add_argument("--smooth",default="0", metavar="smooth",required=False)
    parser.add_argument("--threshhold", default="1", metavar="threshhold", required=True)

    args = parser.parse_args()
    main(args.target, args.models, int(args.episodes), int(args.smooth), float(args.threshhold))
