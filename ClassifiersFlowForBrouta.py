import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from Classifiers import Classifiers
from Utility import Utility
from sklearn.model_selection import KFold
import pandas as pd

class ClassifiersFlowForBrouta:
    utility = None

    def __init__(self):
        ClassifiersFlowForBrouta.utility = Utility()

    def Brouta_feature_selector_caller(self, configs):
        df_distance = ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_distance.xlsx')['Sheet1']
        df_velocity = ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_velocity.xlsx')['Sheet1']

        # In order to remove fisrt column which is file path.
        df_distance = df_distance.iloc[:, 1:]
        df_velocity = df_velocity.iloc[:, 1:]

        df_combined_velocity_distance = pd.concat([df_distance, df_velocity], axis=1)
        df_combined_velocity_distance = df_combined_velocity_distance.iloc[:, 1:]
        # Remove one of the columns with the same name (e.g., remove the first occurrence)
        df_combined_velocity_distance = df_combined_velocity_distance.loc[:, ~df_combined_velocity_distance.columns.duplicated()]

        train_data_distance, test_data_distance, train_labels_distance, test_labels_distance = \
            ClassifiersFlowForBrouta.utility.get_test_train_from_dataframe(
                df_distance, df_distance[configs.is_pd])

        train_data_velocity, test_data_velocity, train_labels_velocity, test_labels_velocity = \
            ClassifiersFlowForBrouta.utility.get_test_train_from_dataframe(
                df_velocity, df_velocity[configs.is_pd])

        train_data_combined, test_data_combined, train_labels_combined, test_labels_combined = \
            ClassifiersFlowForBrouta.utility.get_test_train_from_dataframe(
                df_combined_velocity_distance, df_combined_velocity_distance[configs.is_pd])

        self.Brouta_Selector_Caller(configs, test_data_distance, test_data_velocity, test_labels_distance,
                                    test_labels_velocity, train_data_distance, train_data_velocity,
                                    train_labels_distance,train_labels_velocity,
                                    train_data_combined, test_data_combined,
                                    train_labels_combined, test_labels_combined)
        self.Brouta_classifier_caller()

    def Brouta_classifier_caller(self):
        configs = ClassifiersFlowForBrouta.utility.get_configs()

        df_velocity_test_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_velocity_selected_test_data.xlsx')[
                'Sheet1']
        df_velocity_test_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_velocity_selected_test_labels.xlsx')['Sheet1']
        df_velocity_train_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_velocity_selected_train_data.xlsx')['Sheet1']
        df_velocity_train_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_velocity_selected_train_labels.xlsx')['Sheet1']

        df_distance_test_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_distance_selected_test_data.xlsx')[
                'Sheet1']
        df_distance_test_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_distance_selected_test_labels.xlsx')['Sheet1']
        df_distance_train_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_distance_selected_train_data.xlsx')['Sheet1']
        df_distance_train_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_distance_selected_train_labels.xlsx')['Sheet1']

        df_combined_test_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_combined_selected_test_data.xlsx')[
                'Sheet1']
        df_combined_test_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_combined_selected_test_labels.xlsx')['Sheet1']
        df_combined_train_data = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_combined_selected_train_data.xlsx')['Sheet1']
        df_combined_train_labels = \
            ClassifiersFlowForBrouta.utility.get_dataframe_from_excel(
                'output/output_combined_selected_train_labels.xlsx')['Sheet1']

        print("Distance classifier")
        self.classifier_evaluator(df_velocity_train_data, df_velocity_train_labels[configs.is_pd],
                                  df_velocity_test_data, df_velocity_test_labels[configs.is_pd])
        print("--------------------------------")
        print("Velocity classifier")
        self.classifier_evaluator(df_distance_train_data, df_distance_train_labels[configs.is_pd],
                                  df_distance_test_data, df_distance_test_labels[configs.is_pd])

        print("--------------------------------")
        print("Velocity and Distance classifier combined")
        self.classifier_evaluator(df_combined_train_data, df_combined_train_labels[configs.is_pd],
                                  df_combined_test_data, df_combined_test_labels[configs.is_pd])

    def classifier_evaluator(self, train_data, train_labels, test_data, test_labels):
        # Define the number of folds (k)
        k = 5

        # Initialize the KFold cross-validator
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        knn_classifier_scores = []
        random_forest_classifier_scores = []
        xg_boost_classifier_scores = []
        svm_classifier_scores = []

        scoring_types = ['accuracy', 'f1', 'precision', 'recall']
        # Split the data into k folds and calculate mean y for each fold
        for scoring_type in scoring_types:
            for train_indices, _ in kf.split(train_data):
                fold_train_data, fold_train_labels = train_data.iloc[train_indices], train_labels.iloc[train_indices]
                classifier = Classifiers()
                knn_classifier_scores.append(
                    classifier.knn_classifier_init(fold_train_data, fold_train_labels, test_data,
                                                   test_labels, scoring_type))

                random_forest_classifier_scores.append(classifier.random_forest_classifier_init(fold_train_data,
                                                                                                fold_train_labels,
                                                                                                test_data,
                                                                                                test_labels,
                                                                                                scoring_type))
                xg_boost_classifier_scores.append(
                    classifier.xg_boost_classifier_init(fold_train_data, fold_train_labels,
                                                        test_data, test_labels, scoring_type))
                svm_classifier_scores.append(
                    classifier.svm_model_classifier_init(fold_train_data, fold_train_labels, test_data,
                                                         test_labels, scoring_type))
            print("knn classifier scores of " + scoring_type)
            print(knn_classifier_scores)
            print("random_forest classifier scores of " + scoring_type)
            print(random_forest_classifier_scores)
            print("xg_boost classifier scores of " + scoring_type)
            print(xg_boost_classifier_scores)
            print("svm classifier scores of " + scoring_type)
            print(svm_classifier_scores)
            print("------------------------------------------")
            type_of_result = "Mean of "
            self.print_cross_validation_scores(np.mean(knn_classifier_scores), 'KNN', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.mean(random_forest_classifier_scores), 'random_forest', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.mean(xg_boost_classifier_scores), 'xg_boost', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.mean(svm_classifier_scores), 'svm', scoring_type, type_of_result)
            type_of_result = "Standard deviation of "
            self.print_cross_validation_scores(np.std(knn_classifier_scores), 'KNN', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.std(random_forest_classifier_scores), 'random_forest', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.std(xg_boost_classifier_scores), 'xg_boost', scoring_type, type_of_result)
            self.print_cross_validation_scores(np.std(svm_classifier_scores), 'svm', scoring_type, type_of_result)

            knn_classifier_scores = []
            random_forest_classifier_scores = []
            xg_boost_classifier_scores = []
            svm_classifier_scores = []
            print("----------------------------------")

    def print_cross_validation_scores(self, classifier_scores, model_name, scoring_type, result_type):
        print(result_type + scoring_type + " " + model_name + " :" + str(classifier_scores))

    def Brouta_Selector_Caller(self, configs, test_data_distance, test_data_velocity, test_labels_distance,
                               test_labels_velocity, train_data_distance, train_data_velocity, train_labels_distance,
                               train_labels_velocity,train_data_combined, test_data_combined, train_labels_combined
                               , test_labels_combined):
        print("***************** Starting Boruta *****************")
        boruta_result, selected_features_number_for_distance = self.find_useful_features_in_boruta(train_data_distance,
                                                                                                   configs.is_pd)
        print("Number of useful features to keep distance with Boruta: ", selected_features_number_for_distance)

        "***************************** Distance **********************************"
        Train_data_distance_selected_features = train_data_distance[boruta_result]
        ClassifiersFlowForBrouta.utility.create_feature_scattered_graph(Train_data_distance_selected_features,
                                                                        [1, 2, 3],
                                                                        "Boruta Train data distance selected features")
        Train_data_distance_selected_features.to_excel('output/output_distance_selected_train_data.xlsx')
        Test_data_distance_selected_features = test_data_distance[boruta_result]
        Test_data_distance_selected_features.to_excel('output/output_distance_selected_test_data.xlsx')
        Train_labels_distance_selected_features = train_labels_distance
        Train_labels_distance_selected_features.to_excel('output/output_distance_selected_train_labels.xlsx')
        Test_labels_distance_selected_features = test_labels_distance
        Test_labels_distance_selected_features.to_excel('output/output_distance_selected_test_labels.xlsx')

        "******************************* Velocity *************************************"
        boruta_result_velocity, selected_features_number_for_velocity = self.find_useful_features_in_boruta(
            train_data_velocity,
            configs.is_pd)
        print("Number of useful features to keep for velocity with Boruta: ", selected_features_number_for_velocity)
        Train_data_velocity_selected_features = train_data_velocity[boruta_result_velocity]
        ClassifiersFlowForBrouta.utility.create_feature_scattered_graph(Train_data_velocity_selected_features,
                                                                        [1, 2, 3],
                                                                        "Boruta Train data velocity selected features")
        Train_data_velocity_selected_features.to_excel('output/output_velocity_selected_train_data.xlsx')
        Test_data_velocity_selected_features = test_data_velocity[boruta_result_velocity]
        Test_data_velocity_selected_features.to_excel('output/output_velocity_selected_test_data.xlsx')
        Train_labels_velocity_selected_features = train_labels_velocity
        Train_labels_velocity_selected_features.to_excel('output/output_velocity_selected_train_labels.xlsx')
        Test_labels_velocity_selected_features = test_labels_velocity
        Test_labels_velocity_selected_features.to_excel('output/output_velocity_selected_test_labels.xlsx')

        "******************************* Combined *************************************"
        boruta_result_combined, selected_features_number_for_combined = self.find_useful_features_in_boruta(
            train_data_combined,
            configs.is_pd)
        print("Number of useful features to keep for combined with Boruta: ", selected_features_number_for_combined)
        Train_data_combined_selected_features = train_data_combined[boruta_result_combined]
        ClassifiersFlowForBrouta.utility.create_feature_scattered_graph(Train_data_combined_selected_features,
                                                                        [1, 2, 3],
                                                                        "Boruta Train data velocity selected features")
        Train_data_combined_selected_features.to_excel('output/output_combined_selected_train_data.xlsx')
        Test_data_combined_selected_features = test_data_combined[boruta_result_combined]
        Test_data_combined_selected_features.to_excel('output/output_combined_selected_test_data.xlsx')
        Train_labels_combined_selected_features = train_labels_combined
        Train_labels_combined_selected_features.to_excel('output/output_combined_selected_train_labels.xlsx')
        Test_labels_combined_selected_features = test_labels_combined
        Test_labels_combined_selected_features.to_excel('output/output_combined_selected_test_labels.xlsx')

    def find_useful_features_in_boruta(self, dataframe, target_column):
        # Split the data into features and target
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        # Initialize the Random Forest classifier
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # Initialize the Boruta feature selector
        boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=2, random_state=42)

        # Perform feature selection
        boruta_selector.fit(X.values, y.values)

        # Get the selected features
        selected_features = X.columns[boruta_selector.support_].tolist()

        # result[target_column] = y
        selected_features.append(target_column)
        ## output is features name and target column which is pd
        return selected_features, len(selected_features)
