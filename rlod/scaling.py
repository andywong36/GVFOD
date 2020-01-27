scaling = {
    "robotarm": {
        'time': {'max': 1999, 'min': 0, 'wrap': True, 'divs': "ignore"},
        'direction': {'max': 1, 'min': 0, 'wrap': False, 'divs': "int"},
        'position': {'max': 172.062, 'min': 18.162, 'wrap': False, 'divs': 4},
        'torque': {'max': 0.8102, 'min': -0.7143, 'wrap': False, 'divs': 4},
        'tension': {'max': 245.62001, 'min': 84.43079, 'wrap': False, 'divs': 4}
    },
    "machine": {
        'time': {'max': 255, 'min': 0, 'wrap': True, 'divs': "ignore"},
        'direction': {'max': 1, 'min': 0, 'wrap': False, 'divs': "int"},
        'position_master': {'max': 255.0, 'min': 0.0, 'wrap': True, 'divs': 4},
        'position_slave': {'max': 255.0, 'min': 0.0, 'wrap': True, 'divs': 4},
        'torque_master': {'max': 105.0, 'min': -99.0, 'wrap': False, 'divs': 4},
        'torque_slave': {'max': 96.0, 'min': -115.0, 'wrap': False, 'divs': 4},
        'temperature': {'max': 43.07, 'min': -2.55, 'wrap': False, 'divs': 2},
        'x': {'max': 7.11, 'min': -7.337000000000001, 'wrap': False, 'divs': 2},
        'y': {'max': 18.486, 'min': 3.015, 'wrap': False, 'divs': 2},
        'z': {'max': 7.507999999999999, 'min': -7.507999999999999, 'wrap': False, 'divs': 2}
    }
}


def save(data):
    """
    Saves a dictionary to a json file
    Args:
        data (dict): to save
    Returns:
        None
    """
    import json
    with open('scaling.json', 'w') as outfile:
        json.dump(data, outfile)


def get_max_min(folder, startswith):
    """
    Gets the maximum and minimum of columns of dataframes, across multiple files
    Args:
        folder (str): folder to search in
        startswith (str): only includes files whose names start with this

    Returns:
        max_min (dict), a dictionary of all the colnames, and the max, mins.
    """
    import os
    import pandas as pd

    max_min = dict()
    for filename in os.listdir(folder):
        if filename.startswith(startswith):
            df = pd.read_pickle(folder + '//' + filename)
            for item in df.columns:
                if item not in max_min:
                    max_min[item] = {"max": df[item].max(), "min": df[item].min()}
                else:
                    max_min[item]['max'] = max(df[item].max(), max_min[item]['max'])
                    max_min[item]['min'] = min(df[item].min(), max_min[item]['min'])
    return max_min


if __name__ == "__main__":
    # save(scaling)
    print(get_max_min('data/pickles', 'machine'))