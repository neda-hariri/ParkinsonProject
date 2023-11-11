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


class Classifiers:
    def train_knn_model(self, train_data, train_labels, k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_labels)
        return knn

    def predict_with_knn(self, trained_model, x_test, y_test,scoring_type):
        predictions = trained_model.predict(x_test)
        return self.get_metrics(y_test, predictions,scoring_type)

    def train_xg_boost_model(self, x_train, y_train):
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(x_train, y_train)
        return xgb_model

    def predict_with_xg_boost(self, trained_model, x_test, y_test,scoring_type):
        y_pred = trained_model.predict(x_test)
        return self.get_metrics(y_test, y_pred,scoring_type)

    def train_random_forest_model(self, x_train, y_train):
        # Create Random Forest model
        rf_model = RandomForestClassifier()
        # Train the model
        rf_model.fit(x_train, y_train)
        return rf_model

    def predict_with_random_forest(self, trained_model, x_test, y_test,scoring_type):
        # Make predictions on the test set
        y_pred_rf = trained_model.predict(x_test)

        return self.get_metrics(y_test, y_pred_rf,scoring_type)

    def train_svm_model(self, x_train, y_train):
        # Scale the data for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        # Create SVM model
        svm_model = SVC()
        # Train the model
        svm_model.fit(X_train_scaled, y_train)
        return svm_model, scaler

    def predict_with_svm(self, trained_model, scaler, x_test, y_test,scoring_type):
        x_test_scaled = scaler.transform(x_test)
        y_pred_svm = trained_model.predict(x_test_scaled)
        return  self.get_metrics(y_test, y_pred_svm,scoring_type)


    def get_metrics(self, y_test, predicted_y_test,scoring_type):
        if scoring_type == 'accuracy':
            accuracy = accuracy_score(y_test, predicted_y_test)
            return accuracy
        elif scoring_type == 'f1':
            f1score = f1_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
            return f1score

        elif scoring_type == 'precision':
            precision = precision_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
            return precision

        elif scoring_type == 'recall':
            recall = recall_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
            return recall
        else:
            print("Metric is not defined")

    def knn_classifier_init(self, train_data, train_labels, test_data, test_labels,scoring_type):
        prediction = float('-inf')
        for k in range(1, 20):
            knn_model = self.train_knn_model(train_data, train_labels, k)
            prediction_score = self.predict_with_knn(knn_model, test_data,test_labels,scoring_type)

            if prediction < prediction_score:
                prediction = prediction_score
        return prediction
    def xg_boost_classifier_init(self, train_data, train_labels, test_data, test_labels,scoring_type):
        xg_boost_model = self.train_xg_boost_model(train_data, train_labels)
        return self.predict_with_xg_boost(xg_boost_model, test_data, test_labels,scoring_type)

    def svm_model_classifier_init(self, train_data, train_labels, test_data, test_labels,scoring_type):
        svm_model, scaler = self.train_svm_model(train_data, train_labels)
        return self.predict_with_svm(svm_model, scaler, test_data, test_labels,scoring_type)

    def random_forest_classifier_init(self, train_data, train_labels, test_data, test_labels,scoring_type):
        random_forest_model = self.train_random_forest_model(train_data, train_labels)
        return self.predict_with_random_forest(random_forest_model, test_data, test_labels,scoring_type)

