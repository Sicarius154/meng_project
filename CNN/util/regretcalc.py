import argparse
import matplotlib as mlp
import numpy as np
import pandas as pd
from scipy.integrate import simps #alternative to trapezoid rule

def main(modelname, rounding_value, max_episodes):
    """
    Main entry point for the application
    """
    data = get_data_from_model_csv(modelname) #The data from the log file, this will be a 1D array where the index is the episode number

    if max_episodes != -1: number_episodes = max_episodes
    else: number_episodes = data.shape[0]
    print(f"Getting area for {modelname}...")
    area = get_area_under_plot(data)
    print(f"Got area: {area}")

    ideal_area =  get_area_of_ideal_policy(number_episodes, data)
    print(f"Ideal policy area is: {ideal_area}")

    regret = round(area/ideal_area,rounding_value)
    print(f"Regret is: {regret}")
def get_area_under_plot(data):
    """
    Gets the area underneath the plotted line
    """
    area = np.trapz(data) #use the included trapezium rule function from numpy
    return area
def get_max_reward(data):
    """
    Will find the highest reward achieved and bound the Y axis by this value
    """
    max = np.max(data)
    return max

def get_area_of_ideal_policy(number_eps, data):
    """
    This will get the area of the ideal policy (a perfect rectangle).
    This is calculated as len(data) * get_max_reward()
    """
    return number_eps * get_max_reward(data)

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
        print("Error reading CSV file. Quitting.")

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Calculate regret of a model")
    parser.add_argument("modelname", metavar="modelname")
    parser.add_argument("--round", default="3", metavar="round", required=False)
    parser.add_argument("--episodes", default="-1", metavar="episodes", required=False)

    args = parser.parse_args()
    if args.modelname == "*":
        import glob
        for file in glob.glob('./*.csv'):
            main(file, int(args.round), int(args.episodes))
    else:
        main(args.modelname,int(args.round), int(args.episodes))
