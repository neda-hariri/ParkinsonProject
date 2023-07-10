import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


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
