import numpy as np
from collections import Counter


class KNN:
    def __init__(self, X_train, y_train, distance, K):
        self.X_train = X_train  # Design matrix, where a row is a data
        self.y_train = y_train
        self.nb_points = np.shape(X_train)[0]
        self.K = K
        self.distance = distance  # Distance function

    def predict(self, X):
        # Step 1 : compute distances between X and X_train
        distances = np.array([self.distance(self.X_train[i], X) for i in range(self.nb_points)])

        # Step 2 : find indices of the K-nearest neighbors
        if self.nb_points <= self.K:
            return Counter(self.y_train).most_common(1)[0][0]  # Return most common label
        else:
            idx = np.argpartition(distances, self.K)[:self.K]
            # Get the most common class label among K neighbors
            k_nearest_labels = self.y_train[idx]
            return Counter(k_nearest_labels).most_common(1)[0][0]  # Majority voting

    def accuracy(self, X_test, y_test):
        """Compute accuracy on the test set."""
        nb_test_points = X_test.shape[0]
        correct_predictions = sum(self.predict(X_test[i]) == y_test[i] for i in range(nb_test_points))
        return correct_predictions / nb_test_points
