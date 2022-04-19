import numpy as np
from algorithm.KNN.eucli import *

class KNN():

    def __init__(self, k):
        self.k = k

    def vote(self, neighbor_labels):
        label = max(neighbor_labels, key=neighbor_labels.count)
        return label

    def predict(self, test_data, training_data, training_label):
        predict_label = [0]*len(test_data)
        # Determine the class of each sample
        for i, test_sample in enumerate(test_data):
            test_sample = test_sample.tolist()
            # Sort the training samples by their distance to the test sample and get the K nearest
            index = np.argsort([euclidean_distance(test_sample, x) for x in training_data])[:self.k]
            # get the label list of the k nearest neighboring training data
            k_nearest_neighbors = [training_label[i] for i in index]
            # label test data as the most common labels in neighbors
            predict_label[i] = self.vote(k_nearest_neighbors)

        return predict_label


