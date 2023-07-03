import os
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from Configs import Configs
from FeatureSelector import FeatureSelector

from Utility import Utility

class ThesisPark:
    utility = None

    def __init__(self):
        ThesisPark.utility = Utility()

    def starter(self):
        feature_selector = FeatureSelector()
        feature_selector.feature_selector_caller(self.get_configs(False))

        ## initiator (should input files merged, instead of calculation should only save graphs of input?
        self.initiator(True, False)

    def initiator(self, should_output_merged, is_graph_saved_only):
        if is_graph_saved_only:
            self.data_extraction_calculation(True, is_graph_saved_only)
            self.data_extraction_calculation(False, is_graph_saved_only)
        else:
            if should_output_merged:
                dataframes_result_distance = []
                dataframes_result_velocity = []
                (dataframes_result_distance_pd, dataframes_result_velocity_pd,
                 configs_pd) = self.data_extraction_calculation(True, is_graph_saved_only)
                dataframes_result_distance_pd[configs_pd.is_pd] = 1
                dataframes_result_velocity_pd[configs_pd.is_pd] = 1
                dataframes_result_distance.append(dataframes_result_distance_pd)
                dataframes_result_velocity.append(dataframes_result_velocity_pd)

                (dataframes_result_distance_h, dataframes_result_velocity_h,
                 configs_h) = self.data_extraction_calculation(False, is_graph_saved_only)
                dataframes_result_distance_h[configs_pd.is_pd] = 0
                dataframes_result_velocity_h[configs_pd.is_pd] = 0
                dataframes_result_distance.append(dataframes_result_distance_h)
                dataframes_result_velocity.append(dataframes_result_velocity_h)

                dataframes_result_distance_output = pd.concat(dataframes_result_distance)
                dataframes_result_distance_output_with_selected_features = self.select_features(
                    dataframes_result_distance_output, configs_pd)
                dataframes_result_distance_output_with_selected_features.to_excel('output_distance.xlsx')

                dataframes_result_velocity_output = pd.concat(dataframes_result_velocity)
                dataframes_result_distance_output_with_selected_features = self.select_features(
                    dataframes_result_distance_output, configs_pd)
                dataframes_result_distance_output_with_selected_features.to_excel('output_velocity.xlsx')
            else:
                (dataframes_result_distance_pd, dataframes_result_velocity_pd,
                 configs_pd) = self.data_extraction_calculation(True, is_graph_saved_only)
                (dataframes_result_distance_h, dataframes_result_velocity_h,
                 configs_h) = self.data_extraction_calculation(False, is_graph_saved_only)
                dataframes_result_distance_pd.to_excel('output_distance' + configs_pd.output_extension + '.xlsx')
                dataframes_result_velocity_pd.to_excel('output_velocity' + configs_pd.output_extension + '.xlsx')
                dataframes_result_distance_h.to_excel('output_distance' + configs_h.output_extension + '.xlsx')
                dataframes_result_velocity_h.to_excel('output_velocity' + configs_h.output_extension + '.xlsx')



    def get_configs(self, is_pd):
        if is_pd:
            file_paths = ThesisPark.utility.get_files_in_directory("json/PD/")
            jointPositionTags = 'joint_positions'
            output_extension = '_parkinson'
        else:
            file_paths = ThesisPark.utility.get_files_in_directory("json/HEALTHY/")
            jointPositionTags = 'joints_position'
            output_extension = '_healthy'

        frames = 'frames'
        hands = 'hands'
        timestamp_usec = 'timestamp_usec'
        timestamp = 'timestamp'
        distance = 'Distance'
        velocity = 'Velocity'
        img_output_dir = 'Imgs'
        is_pd = "is_pd"
        return Configs(file_paths, jointPositionTags, frames, hands, timestamp_usec, timestamp, distance, velocity,
                       output_extension, img_output_dir, is_pd)





    def data_extraction_calculation(self, is_input_pd, is_graph_saved_only):
        dataframes_collection_distance = []
        dataframes_collection_velocity = []
        configs = self.get_configs(is_input_pd)

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
                lambda x: ThesisPark.utility.calculate_distance(x[8], x[4])).astype(
                float)

            df = df.rename(columns={configs.joint_Position_tags: configs.distance})

            diff_df = df.diff().dropna()
            df[configs.velocity] = diff_df[configs.distance] / diff_df[configs.timestamp]
            df = df.fillna(df[configs.velocity][1])

            # Extract features
            df["id"] = file_path
            df.set_index('id')
            df['id'] = df['id'].astype(str)

            if is_graph_saved_only:
                self.save_graphs(df, configs, file_path)
            else:
                imputed_features_distance, imputed_features_velocity = self.feature_extraction(df, configs)
                dataframes_collection_distance.append(imputed_features_distance)
                dataframes_collection_velocity.append(imputed_features_velocity)

        if is_graph_saved_only:
            return

        dataframes_result_distance = pd.concat(dataframes_collection_distance)
        dataframes_result_velocity = pd.concat(dataframes_collection_velocity)
        return dataframes_result_distance, dataframes_result_velocity, configs

    def feature_extraction(self, df, configs):

        extracted_features_distance = extract_features(df, column_id="id", \
                                                       column_sort=configs.timestamp, \
                                                       column_value="Distance", \
                                                       disable_progressbar=False, show_warnings=False)

        extracted_features_velocity = extract_features(df, column_id="id", \
                                                       column_sort=configs.timestamp, \
                                                       column_value="Velocity", \
                                                       disable_progressbar=False, show_warnings=False)

        imputed_features_distance = impute(extracted_features_distance)
        imputed_features_velocity = impute(extracted_features_velocity)

        return imputed_features_distance, imputed_features_velocity

    def save_graphs(self, df, configs, json_file_path):
        json_file_name = os.path.basename(json_file_path)

        # Create folders if they don't exist
        modified_uri = json_file_path.replace('\\', '/')
        modified_uri = modified_uri.replace('.json', '')

        distance_dir = os.path.join(configs.img_output_dir, modified_uri, configs.distance)
        velocity_dir = os.path.join(configs.img_output_dir, modified_uri, configs.velocity)
        distance_dir = distance_dir.replace('\\', '/')
        velocity_dir = velocity_dir.replace('\\', '/')
        os.makedirs(distance_dir, exist_ok=True)
        os.makedirs(velocity_dir, exist_ok=True)

        # Plot and save line chart for distance
        distance_chart = plt.figure()
        plt.plot(df[configs.timestamp], df[configs.distance])
        plt.xlabel('Timestamp')
        plt.ylabel('Distance')
        plt.title('Distance vs Timestamp')
        distance_chart_path = os.path.join(distance_dir, 'distance_chart' + json_file_name + '.png')
        plt.savefig(distance_chart_path)
        plt.close(distance_chart)

        # Plot and save line chart for velocity
        velocity_chart = plt.figure()
        plt.plot(df[configs.timestamp], df[configs.velocity])
        plt.xlabel('Timestamp')
        plt.ylabel('Velocity')
        plt.title('Velocity vs Timestamp')
        velocity_chart_path = os.path.join(velocity_dir, 'velocity_chart' + json_file_name + '.png')
        plt.savefig(velocity_chart_path)
        plt.close(velocity_chart)

    def select_features(self, data, configs):
        # Separate the features (X) and the target variable (y)
        X = data.drop(configs.is_pd, axis=1)
        y = data[configs.is_pd]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform feature selection using SelectKBest and f_regression
        selector = SelectKBest(score_func=f_regression, k=3)  # Select the top 3 features
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get the selected feature names
        selected_feature_names = X.columns[selector.get_support(indices=True)].tolist()
        return data[selected_feature_names]



