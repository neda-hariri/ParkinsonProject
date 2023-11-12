import json
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from ClassifiersFlowForBoruta import ClassifiersFlowForBoruta
from ClassifiersPiplineForPCA import ClassifiersPiplineForPCA
from Utility import Utility


class ThesisPark:
    utility = None

    def __init__(self):
        ThesisPark.utility = Utility()

    def starter(self):
        #self.initiator(True,
         #              False)  # To create excel file/graphs (if == true : merge patients , if == true: image save)

        self.PCA_and_classifiers_init()
        self.Boruta_and_classifiers_init()

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
                dataframes_result_distance_output.to_excel('output/output_distance.xlsx')

                dataframes_result_velocity_output = pd.concat(dataframes_result_velocity)
                dataframes_result_velocity_output.to_excel('output/output_velocity.xlsx')
            else:
                (dataframes_result_distance_pd, dataframes_result_velocity_pd,
                 configs_pd) = self.data_extraction_calculation(True, is_graph_saved_only)
                (dataframes_result_distance_h, dataframes_result_velocity_h,
                 configs_h) = self.data_extraction_calculation(False, is_graph_saved_only)
                dataframes_result_distance_pd.to_excel('output_distance' + configs_pd.output_extension + '.xlsx')
                dataframes_result_velocity_pd.to_excel('output_velocity' + configs_pd.output_extension + '.xlsx')
                dataframes_result_distance_h.to_excel('output_distance' + configs_h.output_extension + '.xlsx')
                dataframes_result_velocity_h.to_excel('output_velocity' + configs_h.output_extension + '.xlsx')

    def data_extraction_calculation(self, is_input_pd, is_graph_saved_only):
        dataframes_collection_distance = []
        dataframes_collection_velocity = []
        configs = ThesisPark.utility.get_configs(is_input_pd)

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
                ThesisPark.utility.save_graphs(df, configs, file_path)
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
        extracted_features_distance = extract_features(
            df, column_id="id", column_sort=configs.timestamp, column_value="Distance", disable_progressbar=False,
            show_warnings=False)

        extracted_features_velocity = extract_features(
            df, column_id="id", column_sort=configs.timestamp, column_value="Velocity", disable_progressbar=False,
            show_warnings=False)

        imputed_features_distance = impute(extracted_features_distance)
        imputed_features_velocity = impute(extracted_features_velocity)

        return imputed_features_distance, imputed_features_velocity

    def Boruta_and_classifiers_init(self):
        feature_selector = ClassifiersFlowForBoruta()
        feature_selector.Boruta_feature_selector_caller(ThesisPark.utility.get_configs())

    def PCA_and_classifiers_init(self):
        cpp = ClassifiersPiplineForPCA()
        cpp.PCA_pipeline_with_classifires(ThesisPark.utility.get_configs())
