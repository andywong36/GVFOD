import numpy as np
import pandas as pd


def convert_machine_data(src_dir):
    from pandas.api.types import is_string_dtype

    # time is missing, it will be the 0'th column
    colnames = ("position_master", "position_slave", "torque_master", "torque_slave")

    normal_data = dict()
    abnormal_data = dict()

    normal_data["normal"] = src_dir + r"\99_Normal\normal\_name.csv"
    normal_data['V_normal'] = src_dir + r"\01_VoltageChange\01_VoltageChange(AC200V)\_name.csv"
    normal_data['belt_normal'] = src_dir + r"\04_BeltTension\20170822_BeltTension_normal\_name.csv"

    abnormal_data['misaligned'] = src_dir + r"\00_BeltMeandering\00_BeltMeandering\_name.csv"
    abnormal_data['V_high'] = src_dir + r"\01_VoltageChange\01_VoltageChange(AC220V)\_name.csv"
    abnormal_data['V_low'] = src_dir + r"\01_VoltageChange\01_VoltageChange(AC180V)\_name.csv"
    abnormal_data['install_1'] = src_dir + r"\02_InstallationDistortion\20170724_case1(1600-0200LackOfData)\_name.csv"
    abnormal_data['install_2'] = src_dir + r"\02_InstallationDistortion\20170725_case2\_name.csv"
    abnormal_data['install_3'] = src_dir + r"\02_InstallationDistortion\20170726_case3\_name.csv"
    abnormal_data['install_4'] = src_dir + r"\02_InstallationDistortion\20170727_case4\_name.csv"
    abnormal_data['screw'] = src_dir + r"\03_LoosenScrew\20170803_LoosenScrew(1300-1600LackOfData)\_name.csv"
    abnormal_data['loose'] = src_dir + r"\04_BeltTension\20170824_BeltTension_loose\_name.csv"
    abnormal_data['tight'] = src_dir + r"\04_BeltTension\20170823_BeltTension_tight\_name.csv"
    abnormal_data['noise_power'] = src_dir + r"\05_AddNoise\20170912_AddNoiseToPower\_name.csv"
    abnormal_data['noise_ground'] = src_dir + r"\05_AddNoise\20170913_AddNoiseToGND\_name.csv"
    abnormal_data['noise_signal'] = src_dir + r"\05_AddNoise\20170914_AddNoiseToSignal\_name.csv"

    file_prefixes = ["Resolver_Master", "Resolver_Slave", "Torquetrend_Master", "Torquetrend_Slave"]

    tests_missing_position = ["noise_power", "noise_ground", "noise_signal"]

    for dictionary in [normal_data, abnormal_data]:
        for label in dictionary:
            dataframes = {}
            for src, dest in zip(file_prefixes, colnames):
                if label in tests_missing_position:
                    if "position" in dest:
                        continue
                open_file = dictionary[label].replace("_name", src + "Open")
                open_data = pd.read_csv(open_file, sep=",")
                close_file = dictionary[label].replace("_name", src + "Close")
                close_data = pd.read_csv(close_file, sep=",")
                for data in (open_data, close_data):
                    if is_string_dtype(data[data.columns[0]]):
                        # This is written for the Platform Gate Data
                        # Remove extra characters from index, turn into DateTime
                        stripped = data[data.columns[0]].apply(
                            lambda string: string.strip("(,)"))
                        data[data.columns[0]] = pd.to_datetime(stripped)
                        data.set_index(data.columns[0], inplace=True)
                        data.index.name = None
                open_data.rename(columns=lambda x: "o" + x[-3:], inplace=True)
                close_data.rename(columns=lambda x: "c" + x[-3:], inplace=True)
                assert all(open_data.index == close_data.index)
                data = pd.concat((open_data, close_data), axis=1)
                # print(data.head())
                dataframes[dest] = data

            assert all(dataframes["torque_master"].index == dataframes["torque_slave"].index)

            if not label in tests_missing_position:
                assert all(dataframes["position_master"].index == dataframes["position_slave"].index)
                acc_delta = set([pd.Timedelta(t, 's') for t in [40, 41, 42, 43, 44]])  # acceptable time deltas
                align_times(dataframes, label)

                if not set(dataframes["position_master"].index - dataframes["torque_master"].index).issubset(acc_delta):
                    print("The time deltas do not align:")
                    print(set(dataframes["position_master"].index - dataframes["torque_master"].index))

            # dataframes now contains the data needed to put into a single pandas dataframe
            nrows, ncols = dataframes["torque_master"].shape
            combined_data = pd.DataFrame({"time": np.tile(np.arange(ncols), nrows)})
            combined_data["direction"] = np.tile(
                np.append(np.zeros(ncols // 2), np.ones(ncols // 2)),
                nrows
            ).astype(int)
            if label not in tests_missing_position:
                for col in colnames:
                    combined_data[col] = dataframes[col].values.flatten()
            else:
                for col in ["torque_master", "torque_slave"]:
                    combined_data[col] = dataframes[col].values.flatten()

            # save the dataframe
            pd.to_pickle(combined_data, "pickles\\machine_{}.pkl".format(label))


def convert_machine_temperature_data(src_dir):
    """Reorganizes .csv data files, saves to .pkl file for fast loading

    Args:
        src_dir: the folder containing the .csv files to convert

    Returns:
        None

    """
    from pandas.api.types import is_string_dtype

    abnormal_data = dict()

    # Set source file names
    abnormal_data['cold1'] = src_dir + r"\06_VibrationAndTemperature\171130090918\_name.csv"
    abnormal_data['cold2'] = src_dir + r"\06_VibrationAndTemperature\171130123254\_name.csv"
    abnormal_data['cold3'] = src_dir + r"\06_VibrationAndTemperature\171130155355\_name.csv"
    abnormal_data['hot1'] = src_dir + r"\06_VibrationAndTemperature\171130191459\_name.csv"
    abnormal_data['hot2'] = src_dir + r"\06_VibrationAndTemperature\171130223915\_name.csv"
    abnormal_data['hot3'] = src_dir + r"\06_VibrationAndTemperature\171201020703\_name.csv"
    abnormal_data['hot4'] = src_dir + r"\06_VibrationAndTemperature\171201053444\_name.csv"

    # Set source file sensor names, output column names
    file_prefixes = ["Torquetrend_Master{}", "Torquetrend_Slave{}", "Vibration_{}Temp",
                     "Vibration_{}X", "Vibration_{}Y", "Vibration_{}Z"]
    colnames = ["torque_master", "torque_slave", "temperature", "x", "y", "z"]

    # Read data
    for label in abnormal_data:
        dataframes = {}
        for src, dest in zip(file_prefixes, colnames):
            # Door open
            open_file = abnormal_data[label].replace("_name", src.format("Open"))
            open_data = pd.read_csv(open_file, sep=",")
            # Door close
            close_file = abnormal_data[label].replace("_name", src.format("Close"))
            close_data = pd.read_csv(close_file, sep=",")
            # Processing open and close files...
            for data in (open_data, close_data):
                if is_string_dtype(data[data.columns[0]]):
                    # This is written for the Machine Data
                    # Remove extra characters from index, turn into DateTime
                    stripped = data[data.columns[0]].apply(
                        lambda string: string.strip("(,)"))
                    data[data.columns[0]] = pd.to_datetime(stripped)
                    data.set_index(data.columns[0], inplace=True)
                    data.index.name = None
            # Rename columns
            open_data.rename(columns=lambda x: "o" + x[-3:], inplace=True)
            close_data.rename(columns=lambda x: "c" + x[-3:], inplace=True)
            assert all(open_data.index == close_data.index)
            # Stack the open and close columns side by side
            data = pd.concat((open_data, close_data), axis=1)

            # Cleaning data to handle some input problems.
            if np.any(data.isna(), axis=1).sum() == 1:
                # For dataframes with only a single row containing NaN, fill forward
                data.ffill(axis=1, inplace=True)

            # Save results
            dataframes[dest] = data

            # Error checking
            if data.isnull().sum().sum() > 0:
                pass

        # Exception handling
        if label == "hot1":
            for key in ["torque_master", "torque_slave"]:
                dataframes[key] = dataframes[key].iloc[2:, :]

        for key in dataframes:
            assert all(dataframes["torque_master"].index == dataframes[key].index)

        nrows, ncols = dataframes["torque_master"].shape
        combined_data = pd.DataFrame({"time": np.tile(np.arange(ncols), nrows)})
        combined_data["direction"] = np.tile(
            np.append(np.zeros(ncols // 2), np.ones(ncols // 2)),
            nrows
        ).astype(int)
        for col in colnames:
            if dataframes[col].shape[1] != dataframes['torque_master'].shape[1]:
                print("Resampling {}".format(col))
                J = np.floor(np.linspace(0, dataframes[col].shape[1] - 1,
                                         dataframes['torque_master'].shape[1])).astype(int)
                combined_data[col] = dataframes[col].values[:, J].flatten()
            else:
                combined_data[col] = dataframes[col].values.flatten()

        # save the dataframe
        pd.to_pickle(combined_data, "pickles\\machine_{}.pkl".format(label))


def align_times(dataframes, label):
    pos_idx = 0
    tor_idx = 0
    pos_max = len(dataframes["position_master"]) - 1
    tor_max = len(dataframes["torque_master"]) - 1

    pos_ndrops = 0
    tor_ndrops = 0

    accept_offsets_seconds = [40, 41, 42, 43, 44]
    if label == "V_normal":
        accept_offsets_seconds.extend([64])

    tor_times = dataframes["torque_master"].index
    while pos_idx < pos_max:
        pos_times = dataframes["position_master"].index.copy()
        if any(pos_times[pos_idx] - pd.Timedelta(t, 's') in tor_times for t in accept_offsets_seconds):
            pos_idx += 1
        else:
            best = np.argsort(np.abs(pos_times[pos_idx] - tor_times))[:4]
            print("Could not find matching time for position {} in torques, {}".format(pos_times[pos_idx], label))
            print("\t The best offset was {}, {}, {}, {}".format(*[pos_times[pos_idx] - tor_times[b] for b in best]))
            dataframes["position_master"].drop(pos_times[pos_idx], inplace=True)
            dataframes["position_slave"].drop(pos_times[pos_idx], inplace=True)
            pos_ndrops += 1
            pos_max -= 1

    pos_times = dataframes["position_master"].index
    while tor_idx < tor_max:
        tor_times = dataframes["torque_master"].index.copy()
        if any(tor_times[tor_idx] + pd.Timedelta(t, 's') in pos_times for t in accept_offsets_seconds):
            tor_idx += 1
        else:
            best = np.argsort(np.abs(pos_times - tor_times[tor_idx]))[:4]
            print("Could not find matching time for torque {} in positions, {}".format(tor_times[tor_idx], label))
            print("\t The best offset was {}, {}, {}, {}".format(*[pos_times[b] - tor_times[tor_idx] for b in best]))
            dataframes["torque_master"].drop(tor_times[tor_idx], inplace=True)
            dataframes["torque_slave"].drop(tor_times[tor_idx], inplace=True)
            tor_ndrops += 1
            tor_max -= 1

    return 0



if __name__ == "__main__":
    import sys
    machine_data_dir = sys.argv[1]
    # convert_machine_data(machine_data_dir)
    # convert_machine_temperature_data(machine_data_dir)
