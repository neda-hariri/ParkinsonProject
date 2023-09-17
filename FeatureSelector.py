from Utility import Utility
from sklearn.decomposition import PCA
from boruta import BorutaPy
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn.neighbors

class FeatureSelector:
    utility =None
    def __init__(self):
        FeatureSelector.utility=Utility()

    def feature_selector_caller(self, configs):

        df_distance = FeatureSelector.utility.get_dataframe_from_excel('output/output_distance.xlsx')['Sheet1']
        df_velocity = FeatureSelector.utility.get_dataframe_from_excel('output/output_velocity.xlsx')['Sheet1']

        df_distance = df_distance.iloc[:, 1:]
        df_velocity = df_velocity.iloc[:, 1:]

        print("***************** Starting PCA *****************")
        num_useful_features_df_distance = self.find_useful_features(df_distance)
        num_useful_features_df_velocity = self.find_useful_features(df_velocity)
        print("Number of useful features to keep distance: ", num_useful_features_df_distance)
        print("Number of useful features to keep for velocity: ", num_useful_features_df_velocity)
        print("***************** Starting Boruta *****************")
        boruta_result, selected_feature_num_dis = self.boruta_feature_selection(df_distance, configs.is_pd)
        print("Number of useful features to keep distance: ", selected_feature_num_dis)

        boruta_result.to_excel('output/output_distance_selected.xlsx')

        boruta_result_vel, selected_feature_num_vel = self.boruta_feature_selection(df_velocity, configs.is_pd)
        print("Number of useful features to keep for velocity: ", selected_feature_num_vel)
        boruta_result_vel.to_excel('output/output_velocity_selected.xlsx')

    def find_useful_features(self, dataframe, threshold_variance=0.90):
        """
        Perform PCA on the input dataframe and find the number of useful features to keep based on the given threshold.

        Parameters:
        dataframe (pd.DataFrame): Input dataframe with features.
        threshold_variance (float): Threshold value for cumulative explained variance. Defaults to 0.95.

        Returns:
        int: Number of useful features to keep.
        """
        # Standardize the data (optional but usually recommended for PCA)
        standardized_data = (dataframe - dataframe.mean()) / dataframe.std()
        standardized_data = standardized_data.dropna(axis=1)

        imputer = SimpleImputer(strategy='mean')
        imputer.fit(standardized_data)
        df_filled = imputer.transform(standardized_data)
        standardized_data = pd.DataFrame(df_filled, columns=standardized_data.columns)

        # Initialize the PCA model
        pca = PCA()

        # Fit the model on the standardized data
        pca.fit(standardized_data)

        # Compute the cumulative explained variance
        explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

        # Find the number of components that explain the specified threshold variance
        num_useful_features = np.argmax(explained_variance_ratio_cumsum >= threshold_variance) + 1

        return num_useful_features

    def boruta_feature_selection(self, dataframe, target_column):
        """
        Perform feature selection using Boruta algorithm.

        Parameters:
        dataframe (pd.DataFrame): Input dataframe with features and target column.
        target_column (str): Name of the target column.

        Returns:
        pd.DataFrame: DataFrame containing selected features and their importance scores.
        """
        # Split the data into features and target
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        # Initialize the Random Forest classifier
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=4)

        # Initialize the Boruta feature selector
        boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=2, random_state=1)

        # Perform feature selection
        boruta_selector.fit(X.values, y.values)

        # Get the selected features
        selected_features = X.columns[boruta_selector.support_].tolist()

        # Get the feature importance scores
        feature_importance = boruta_selector.ranking_
        result = X[selected_features]
        # Create a DataFrame with selected features and their importance scores
        # result_df = pd.DataFrame({'Feature': X, 'Importance': feature_importance})
        # result_df = result_df.sort_values(by='Importance', ascending=True).reset_index(drop=True)

        result[target_column] = y
        return result, len(selected_features)
