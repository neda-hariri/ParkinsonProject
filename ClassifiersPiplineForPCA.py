import ThesisPark
from Utility import Utility
from sklearn.decomposition import PCA
from boruta import BorutaPy
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier


class ClassifiersPiplineForPCA:
    utility = None

    def __init__(self):
        ClassifiersPiplineForPCA.utility = Utility()

    def PCA_pipeline_with_classifires(self, configs):
        df_distance = ClassifiersPiplineForPCA.utility.get_dataframe_from_excel('output/output_distance.xlsx')['Sheet1']
        df_velocity = ClassifiersPiplineForPCA.utility.get_dataframe_from_excel('output/output_velocity.xlsx')['Sheet1']

        df_distance = df_distance.iloc[:, 1:]
        df_velocity = df_velocity.iloc[:, 1:]

        train_data_distance, test_data_distance, train_labels_distance, test_labels_distance = ClassifiersPiplineForPCA.utility.get_test_train_from_dataframe(
            df_distance,
            df_distance[configs.is_pd])
        train_data_velocity, test_data_velocity, train_labels_velocity, test_labels_velocity = ClassifiersPiplineForPCA.utility.get_test_train_from_dataframe(
            df_velocity,
            df_velocity[configs.is_pd])

        self.PCA_selector_caller(test_data_distance, test_labels_distance,
                                 train_data_distance, train_labels_distance, configs.distance)

        self.PCA_selector_caller(test_data_velocity, test_labels_velocity, train_data_velocity,
                                 train_labels_velocity, configs.velocity)

    def PCA_selector_caller(self, test_data, test_labels, train_data, train_labels, Evaluation_label):
        print("***************** Starting PCA for " + Evaluation_label + " *****************")
        PCA_threshold = self.draw_chart_and_get_threshold_for_PCA(train_data)

        # Initialize the PCA model
        pca = PCA(PCA_threshold)
        knn_classifier = KNeighborsClassifier()
        xgb_classifier = xgb.XGBClassifier()
        rf_classifier = RandomForestClassifier()
        svm_classifier = SVC()

        # Create pipelines with PCA and different classifiers
        pca_svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('svm', svm_classifier)
        ])

        pca_rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('rf', rf_classifier)
        ])

        pca_xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', pca),
            ('rf', xgb_classifier)
        ])

        pca_knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Optional: Standardize features
            ('pca', pca),
            ('rf', knn_classifier)
        ])

        pca_svm_pipeline.fit(train_data, train_labels)
        pca_rf_pipeline.fit(train_data, train_labels)
        pca_xgb_pipeline.fit(train_data, train_labels)
        pca_knn_pipeline.fit(train_data, train_labels)

        scoring_type = 'accuracy'
        number_of_cv_splits = 5
        print('PCA svc pipeline results using ' + scoring_type + ' for scoring')
        print(cross_val_score(pca_svm_pipeline, test_data, test_labels, cv=number_of_cv_splits,
                              scoring=scoring_type))
        print('PCA Random Forest Classifier pipeline results using ' + scoring_type + ' for scoring')
        print(cross_val_score(pca_rf_pipeline, test_data, test_labels, cv=number_of_cv_splits,
                              scoring=scoring_type))
        print('PCA XGBClassifier pipeline results using ' + scoring_type + ' for scoring')
        print(cross_val_score(pca_xgb_pipeline, test_data, test_labels, cv=number_of_cv_splits,
                              scoring=scoring_type))
        print('PCA KNN pipeline results using ' + scoring_type + ' for scoring')
        print(cross_val_score(pca_knn_pipeline, test_data, test_labels, cv=number_of_cv_splits,
                              scoring=scoring_type))

        return None

    def draw_chart_and_get_threshold_for_PCA(self, dataframe):
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

        x = range(0, explained_variance_ratio_cumsum.shape[0])
        first_derivative = np.gradient(pca.explained_variance_ratio_, x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, first_derivative, label='First Derivative')
        Q1 = np.percentile(first_derivative, 25)
        Q3 = np.percentile(first_derivative, 75)
        IQR = Q3 - Q1
        threshold = np.argmax(np.abs(first_derivative) < IQR)
        plt.axvline(threshold, color='r', linestyle='--', label='Constant Value')
        plt.text(threshold, 0, f'Constant Value: {threshold}', rotation=90, va='bottom')

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Second Derivative of a Series')
        plt.legend()

        ## add plot for Individual and Cumulative explained variance
        plt.bar(x, pca.explained_variance_ratio_, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(x, explained_variance_ratio_cumsum, where='mid',
                 label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        threshold_for_variance_ratio = threshold / len(pca.explained_variance_ratio_)
        return threshold_for_variance_ratio