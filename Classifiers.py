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

class Classifiers:
    def train_knn_model(self, train_data, train_labels, k):
        # Instantiate KNN model
        knn = KNeighborsRegressor(n_neighbors=k)
        # Fit the model with training data
        knn.fit(train_data, train_labels)
        # Return the trained model
        return knn

    def predict_with_knn(self, trained_model, new_data):
        # Use the trained model to find nearest neighbors and predict
        predictions = trained_model.predict(new_data)
        # Return the predictions
        return predictions

    def train_xg_boost_model(self, x_train, y_train):
        # Create XGBoost model
        xgb_model = xgb.XGBClassifier()
        # Train the model
        xgb_model.fit(x_train, y_train)
        return xgb_model

    def predict_with_xg_boost(self, trained_model, x_test, y_test):
        # Make predictions on the test set
        y_pred = trained_model.predict(x_test)

        accuracy, f1score, precision, recall = self.get_metrics(y_test, y_pred)
        print("XG_Boost Accuracy:", accuracy)
        print("XG_Boost F1score:", f1score)
        print("XG_Boost Precision:", precision)
        print("XG_Boost Recall:", recall)

    def train_random_forest_model(self, x_train, y_train):
        # Create Random Forest model
        rf_model = RandomForestClassifier()
        # Train the model
        rf_model.fit(x_train, y_train)
        return rf_model

    def predict_with_random_forest(self, trained_model, x_test, y_test):
        # Make predictions on the test set
        y_pred_rf = trained_model.predict(x_test)

        accuracy, f1score, precision, recall = self.get_metrics(y_test, y_pred_rf)
        print("Random Forest Accuracy:", accuracy)
        print("Random Forest F1score:", f1score)
        print("Random Forest Precision:", precision)
        print("Random Forest Recall:", recall)

    def train_svm_model(self, x_train, y_train):
        # Scale the data for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(x_train)
        # Create SVM model
        svm_model = SVC()
        # Train the model
        svm_model.fit(X_train_scaled, y_train)
        return svm_model, scaler

    def predict_with_svm(self, trained_model, scaler, x_test, y_test):
        x_test_scaled = scaler.transform(x_test)
        y_pred_svm = trained_model.predict(x_test_scaled)

        accuracy, f1score, precision, recall = self.get_metrics(y_test, y_pred_svm)
        print("SVM Accuracy:", accuracy)
        print("SVM F1score:", f1score)
        print("SVM Precision:", precision)
        print("SVM Recall:", recall)

    def get_metrics(self, y_test, predicted_y_test):
        accuracy = accuracy_score(y_test, predicted_y_test)
        f1score = f1_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
        precision = precision_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
        recall = recall_score(y_test, predicted_y_test, average='macro', zero_division=0.0)
        return accuracy, f1score, precision, recall

    def get_metrics_with_cross_validation(self,model,X_values,y_values):
        print("CrossValidation: ")
        print(cross_val_score(model, X_values, y_values, cv=6))

    def knn_classifier_init(self, train_data, train_labels, test_data, test_labels):
        print("******************* Starting KNN Classifier ********************")
        accuracy = float('-inf')
        i_max_accuracy = -1;
        f1score = 0
        precision = 0
        recall = 0
        for i in range(1, 20):
            knn_model = self.train_knn_model(train_data, train_labels, i)
            self.get_metrics_with_cross_validation(knn_model, test_data, test_labels)
            prediction = self.predict_with_knn(knn_model, test_data)

            test_lb = np.floor((test_labels.values * 1000))
            pr_lb = np.floor(prediction * 1000)
            accuracy_i = accuracy_score(test_lb, pr_lb)
            f1score_i = f1_score(test_lb, pr_lb, average='macro', zero_division=0.0)
            precision_i = precision_score(test_lb, pr_lb, average='macro', zero_division=0.0)
            recall_i = recall_score(test_lb, pr_lb, average='macro', zero_division=0.0)
            if accuracy < accuracy_i:
                accuracy = accuracy_i
                f1score = f1score_i
                precision = precision_i
                recall = recall_i
                i_max_accuracy = i

        print("KNN K: " + str(i_max_accuracy))
        print("KNN Accuracy: " + str(accuracy))
        print("KNN F1score: " + str(f1score))
        print("KNN Precision: " + str(precision))
        print("KNN Recall: " + str(recall))

    def xg_boost_classifier_init(self, train_data, train_labels, test_data, test_labels):
        print("******************* Starting XGBoost Classifier ********************")
        xg_boost_model = self.train_xg_boost_model(train_data, train_labels)
        self.predict_with_xg_boost(xg_boost_model, test_data, test_labels)
        self.get_metrics_with_cross_validation(xg_boost_model,test_data, test_labels)

    def svm_model_classifier_init(self, train_data, train_labels, test_data, test_labels):
        print("******************* Starting SVM Classifier ********************")
        svm_model, scaler = self.train_svm_model(train_data, train_labels)
        self.predict_with_svm(svm_model, scaler, test_data, test_labels)
        self.get_metrics_with_cross_validation(svm_model,test_data, test_labels)

    def random_forest_classifier_init(self, train_data, train_labels, test_data, test_labels):
        print("******************* Starting Random Forest Classifier ********************")
        random_forest_model = self.train_random_forest_model(train_data, train_labels)
        self.predict_with_random_forest(random_forest_model, test_data, test_labels)
        self.get_metrics_with_cross_validation(random_forest_model,test_data, test_labels)