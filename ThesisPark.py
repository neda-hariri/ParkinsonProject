import os

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import openpyxl
import scipy
from Configs import Configs


class ThesisPark:
    def starter(self):
        self.initiator(True)

    def initiator(self, should_output_merged):
        if should_output_merged:
            dataframes_result_distance = []
            dataframes_result_velocity = []
            (dataframes_result_distance_pd, dataframes_result_velocity_pd, configs_pd) = self.calculation(True)
            dataframes_result_distance_pd["is_pd"] = 1
            dataframes_result_velocity_pd["is_pd"] = 1
            dataframes_result_distance.append(dataframes_result_distance_pd)
            dataframes_result_velocity.append(dataframes_result_velocity_pd)

            (dataframes_result_distance_h, dataframes_result_velocity_h, configs_h) = self.calculation(False)
            dataframes_result_distance_h["is_pd"] = 0
            dataframes_result_velocity_h["is_pd"] = 0
            dataframes_result_distance.append(dataframes_result_distance_h)
            dataframes_result_velocity.append(dataframes_result_velocity_h)

            dataframes_result_distance_output = pd.concat(dataframes_result_distance)
            dataframes_result_distance_output.to_excel('output_distance.xlsx')

            dataframes_result_velocity_output = pd.concat(dataframes_result_velocity)
            dataframes_result_velocity_output.to_excel('output_velocity.xlsx')
        else:
            (dataframes_result_distance_pd, dataframes_result_velocity_pd, configs_pd) = self.calculation(True)
            (dataframes_result_distance_h, dataframes_result_velocity_h, configs_h) = self.calculation(False)
            dataframes_result_distance_pd.to_excel('output_distance' + configs_pd.output_extension + '.xlsx')
            dataframes_result_velocity_pd.to_excel('output_velocity' + configs_pd.output_extension + '.xlsx')
            dataframes_result_distance_h.to_excel('output_distance' + configs_h.output_extension + '.xlsx')
            dataframes_result_velocity_h.to_excel('output_velocity' + configs_h.output_extension + '.xlsx')

    @staticmethod
    def calculate_distance(point1, point2):
        (x1, y1, z1) = point1
        (x2, y2, z2) = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def get_files_in_directory(self, directory):
        file_list = []
        for root, directories, files in os.walk(directory):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def getConfigs(self, is_pd):
        if is_pd:
            file_paths = self.get_files_in_directory("json/PD/")
            jointPositionTags = 'joint_positions'
            output_extension = '_parkinson'
        else:
            file_paths = self.get_files_in_directory("json/HEALTHY/")
            jointPositionTags = 'joints_position'
            output_extension = '_healthy'

        frames = 'frames'
        hands = 'hands'
        timestamp_usec = 'timestamp_usec'
        timestamp = 'timestamp'
        distance = 'Distance'
        velocity = 'Velocity'

        return Configs(file_paths, jointPositionTags, frames, hands, timestamp_usec, timestamp, distance, velocity,
                       output_extension)

    def calculation(self, is_input_pd):
        dataframes_collection_distance = []
        dataframes_collection_velocity = []

        configs = self.getConfigs(is_input_pd)

        for file_path in configs.file_paths:
            with open(file_path) as file:
                data = json.load(file)

            # Extract the relevant data from the JSON structure
            frames = data[configs.frames]

            # Extract the data for each hand into a list of dictionaries
            hands_data = []
            timestamp_data = []
            for frame in frames:
                frame_data = frame[configs.hands]
                timestamp = frame[configs.timestamp_usec]
                hands_data.extend(frame_data)
                timestamp_data.append(timestamp)

            # Create a DataFrame from the hands data
            df = pd.DataFrame(hands_data)

            # Extract the specific columns of interest
            df = df[[configs.joint_Position_tags]]
            df[configs.timestamp] = timestamp_data
            # Perform the required computations on the columns
            df[configs.joint_Position_tags] = df[configs.joint_Position_tags].apply(
                lambda x: self.calculate_distance(x[8], x[4])).astype(
                float)

            df = df.rename(columns={configs.joint_Position_tags: configs.distance})

            diff_df = df.diff().dropna()
            df[configs.velocity] = diff_df[configs.distance] / diff_df[configs.timestamp]
            df = df.fillna(df[configs.velocity][1])

            # Extract features
            # df.insert(0, 'id', range(0, len(df)))
            df["id"] = file_path
            df.set_index('id')
            df['id'] = df['id'].astype(str)

            extracted_features_distance = extract_features(df, column_id="id", \
                                                           column_sort=configs.timestamp, \
                                                           column_value="Distance", \
                                                           disable_progressbar=False, show_warnings=False)

            extracted_features_velocity = extract_features(df, column_id="id", \
                                                           column_sort=configs.timestamp, \
                                                           column_value="Velocity", \
                                                           disable_progressbar=False, show_warnings=False)

            imputed_features_distance = impute(extracted_features_distance)
            dataframes_collection_distance.append(imputed_features_distance)

            imputed_features_velocity = impute(extracted_features_velocity)
            dataframes_collection_velocity.append(imputed_features_velocity)

        dataframes_result_distance = pd.concat(dataframes_collection_distance)
        dataframes_result_velocity = pd.concat(dataframes_collection_velocity)

        return dataframes_result_distance, dataframes_result_velocity, configs

        # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        #
        # axs[0, 0].plot(df[configs.distance].to_frame())
        # axs[0, 0].set_xlabel('Sample')
        # axs[0, 0].set_ylabel('Distance in mm')
        # axs[0, 0].set_title('Parkinson Distances')
        #

        # axs[1, 0].plot(df[configs.velocity].to_frame())
        # axs[1, 0].set_xlabel('Sample')
        # axs[1, 0].set_ylabel('Velocity in mm/timestamp')
        # axs[1, 0].set_title('Parkinson Velocities')

    # axs[1, 1].plot(extracted_features_velocity['Velocity__mean'].to_numpy())
    # axs[1, 1].set_xlabel('Sample')
    # axs[1, 1].set_ylabel('Velocity__mean in mm/timestamp')
    # axs[1, 1].set_title('Parkinson Velocity__mean')

    #   plt.show()
