import numpy as np


class KNN:
    def __init__(self, X_train, y_train, distance, K):
        super(KNN, self).__init__()
        self.X_train = X_train  # Design matrix, where a row is a data
        self.y_train = y_train
        self.nb_points = np.shape(X_train)[0]
        self.K = K
        self.distance = distance  # self.distance is a function

    def predict(self, X):

        # Step 1 : compute euclidean distances between X and X_train
        distances = np.zeros(self.nb_points)
        for i in range(self.nb_points):
            distances[i] = self.distance(self.X_train[i], X)

        # Step 2 : find the indices of the K-nearest pts
        if self.nb_points <= self.K:
            # Step 3 : compute predict as an average of the K nearest points
            return np.mean(self.y_train)
        else:
            # Step 3 : compute predict as an average of the K nearest points
            idx = np.argpartition(distances, self.K)
            return np.mean(self.y_train[idx[:self.K]])


    def RMSE(self, X_test, y_test):
        nb_test_points = X_test.shape[0]
        squared_sum = 0
        for i in range(nb_test_points):
            squared_sum = squared_sum + ((self.predict(X_test[i]) - y_test[i])**2)/nb_test_points
        return np.sqrt(squared_sum)

