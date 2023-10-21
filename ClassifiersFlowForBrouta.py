from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from Classifiers import Classifiers
from Utility import Utility
from sklearn.model_selection import KFold


class ClassifiersFlowForBrouta:
    utility = None

    def __init__(self):
        ClassifiersFlowForBrouta.utility = Utility()

    def Brouta_feature_selector_caller(self, configs):
        df_distance = ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_distance.xlsx')['Sheet1']
        df_velocity = ClassifiersFlowForBrouta.utility.get_dataframe_from_excel('output/output_velocity.xlsx')['Sheet1']

        df_distance = df_distance.iloc[:, 1:]
        df_velocity = df_velocity.iloc[:, 1:]

        train_data_distance, test_data_distance, train_labels_distance, test_labels_distance = \
            ClassifiersFlowForBrouta.utility.get_test_train_from_dataframe(
                df_distance, df_distance[configs.is_pd])
        train_data_velocity, test_data_velocity, train_labels_velocity, test_labels_velocity = \
            ClassifiersFlowForBrouta.utility.get_test_train_from_dataframe(
                df_velocity, df_velocity[configs.is_pd])

        self.Brouta_Selector_Caller(configs, test_data_distance, test_data_velocity, test_labels_distance,
                                    test_labels_velocity, train_data_distance, train_data_velocity,
                                    train_labels_distance,
                                    train_labels_velocity)
        self.Brouta_classifier_caller()

    def Brouta_classifier_caller(self):
        configs = ClassifiersFlowForBrouta.utility.get_configs(False)

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

        print("Distance classifier")
        self.classifier_evaluator(df_velocity_train_data, df_velocity_train_labels[configs.is_pd],
                                  df_velocity_test_data, df_velocity_test_labels[configs.is_pd])
        print("--------------------------------")
        print("Velocity classifier")
        self.classifier_evaluator(df_distance_train_data, df_distance_train_labels[configs.is_pd],
                                  df_distance_test_data, df_distance_test_labels[configs.is_pd])

    def classifier_evaluator(self, train_data, train_labels, test_data, test_labels):
        # Define the number of folds (k)
        k = 3

        # Initialize the KFold cross-validator
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        # Split the data into k folds and calculate mean y for each fold
        for train_indices, _ in kf.split(train_data):
            fold_train_data, fold_train_labels = train_data.iloc[train_indices], train_labels.iloc[train_indices]
            classifier = Classifiers()
            knn_classifier_scores = classifier.knn_classifier_init(fold_train_data, fold_train_labels, test_data,
                                                                   test_labels)

            random_forest_classifier_scores = classifier.random_forest_classifier_init(fold_train_data,
                                                                                       fold_train_labels, test_data,
                                                                                       test_labels)
            xg_boost_classifier_scores = classifier.xg_boost_classifier_init(fold_train_data, fold_train_labels,
                                                                             test_data, test_labels)
            svm_classifier_scores = classifier.svm_model_classifier_init(fold_train_data, fold_train_labels, test_data,
                                                                         test_labels)

            self.print_cross_validation_scores(knn_classifier_scores, 'KNN')
            self.print_cross_validation_scores(random_forest_classifier_scores, 'random_forest')
            self.print_cross_validation_scores(xg_boost_classifier_scores, 'xg_boost')
            self.print_cross_validation_scores(svm_classifier_scores, 'svm')

    def print_cross_validation_scores(self, classifier_scores, model_name):
        mean_accuracy = classifier_scores.mean()
        std_accuracy = classifier_scores.std()
        print("Mean Accuracy " + model_name + " :" + str(mean_accuracy))
        print("STD Accuracy " + model_name + " :" + str(std_accuracy))

    def Brouta_Selector_Caller(self, configs, test_data_distance, test_data_velocity, test_labels_distance,
                               test_labels_velocity, train_data_distance, train_data_velocity, train_labels_distance,
                               train_labels_velocity):
        print("***************** Starting Boruta *****************")
        boruta_result, selected_feature_num_dis = self.find_useful_features_in_boruta(train_data_distance,
                                                                                      configs.is_pd)
        print("Number of useful features to keep distance with Boruta: ", selected_feature_num_dis)
        Train_data_distance_selected_features = train_data_distance[boruta_result]
        Train_data_distance_selected_features.to_excel('output/output_distance_selected_train_data.xlsx')
        Test_data_distance_selected_features = test_data_distance[boruta_result]
        Test_data_distance_selected_features.to_excel('output/output_distance_selected_test_data.xlsx')
        Train_labels_distance_selected_features = train_labels_distance
        Train_labels_distance_selected_features.to_excel('output/output_distance_selected_train_labels.xlsx')
        Test_labels_distance_selected_features = test_labels_distance
        Test_labels_distance_selected_features.to_excel('output/output_distance_selected_test_labels.xlsx')
        boruta_result_vel, selected_feature_num_vel = self.find_useful_features_in_boruta(train_data_velocity,
                                                                                          configs.is_pd)
        print("Number of useful features to keep for velocity with Boruta: ", selected_feature_num_vel)
        Train_data_velocity_selected_features = train_data_velocity[boruta_result_vel]
        Train_data_velocity_selected_features.to_excel('output/output_velocity_selected_train_data.xlsx')
        Test_data_velocity_selected_features = test_data_velocity[boruta_result_vel]
        Test_data_velocity_selected_features.to_excel('output/output_velocity_selected_test_data.xlsx')
        Train_labels_velocity_selected_features = train_labels_velocity
        Train_labels_velocity_selected_features.to_excel('output/output_velocity_selected_train_labels.xlsx')
        Test_labels_velocity_selected_features = test_labels_velocity
        Test_labels_velocity_selected_features.to_excel('output/output_velocity_selected_test_labels.xlsx')

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