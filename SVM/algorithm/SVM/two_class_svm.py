import numpy as np
import cvxopt
from sklearn import preprocessing

Minimum_Multiplier_Value = 1e-7


class SVM(object):
    def __init__(self, C, kernel, sigma = None):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma

    def train(self, X, y):
        # normalize the training features X
        X = preprocessing.scale(X)
        lagrange_multipliers = self.compute_lagrange_multipliers(X, y)
        print('lagrange_multipliers',lagrange_multipliers)
        return self.construct_SVMpredictor(X, y, lagrange_multipliers)

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        if not self.sigma:
            self.sigma = 1 / n_features

    def gram_matrix(self, X):
        n_samples, n_features = X.shape
        kernel_matrix = np.zeros((n_samples, n_samples))
        # print('self.kernel1', self.kernel)
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                kernel_matrix[i, j] = self.kernel(x_i, x_j)

        return kernel_matrix

    # using cvxopt.solvers.qp to solve for alpha_i vector,which is lagrange multiplier
    def compute_lagrange_multipliers(self, X, y):
        n_samples, n_features = X.shape
        kernel_matrix = self.gram_matrix(X)

        # cvxopt.solvers.qp solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx <= h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -alpha_i <= 0
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        # alpha_i <= C, C is introduced as a penalty factor so that we could have a soft-margin classifier
        G_soft = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_soft = cvxopt.matrix(np.ones(n_samples) * self.C)
        # combine both G
        G = cvxopt.matrix(np.vstack((G_std, G_soft)))
        h = cvxopt.matrix(np.vstack((h_std, h_soft)))

        # sum of alpha_i*y_i = 0
        A = cvxopt.matrix(y, (1, n_samples),'d')
        b = cvxopt.matrix(0.0)

        question = cvxopt.solvers.qp(P, q, G, h, A, b)
        solution = np.ravel(question['x'])

        # alpha_i vector as lagrange multipliers is solved
        return solution

    def construct_SVMpredictor(self, X, y, lagrange_multipliers):
        # first we need to calculate the W and the bias
        # since only the lagrange_multipliers of support vectors matter
        # also, lagrange_multipliers of support vectors are greater than 0, so we pick a similar value
        support_vector_index = lagrange_multipliers > Minimum_Multiplier_Value
        # print('support_vector_index',support_vector_index)
        useful_lagrange_multipliers = lagrange_multipliers[support_vector_index]
        support_vectors = X[support_vector_index]
        support_vector_labels = y[support_vector_index]
        print('support_vector_labels',support_vector_labels)

        # calculate bias
        # bias = y_s - sum[alpha_i*y_i*kernel(x_i*x_s)]
        # x_i is all support vectors, x_s is just any one of support vectors
        bias = support_vector_labels[-1]
        for i in range(len(useful_lagrange_multipliers)):
            bias -= useful_lagrange_multipliers[i] * support_vector_labels[i] * self.kernel(support_vectors[i], support_vectors[-1])

        #  get a SVMPredictor that could use test data to predict their labels
        return SVMPredictor(
            kernel=self.kernel,
            bias=bias,
            useful_lagrange_multipliers=useful_lagrange_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)


class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 useful_lagrange_multipliers,
                 support_vectors,
                 support_vector_labels):
        self.kernel = kernel
        self.bias = bias
        self.useful_lagrange_multipliers = useful_lagrange_multipliers
        self.support_vectors = support_vectors
        self.support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(useful_lagrange_multipliers) == len(support_vector_labels)
        print("Bias: %s", self.bias)
        print("useful_lagrange_multipliers: %s", len(self.useful_lagrange_multipliers))
        print("Support vectors: %s", len(self.support_vectors))
        print("Support vector labels: %s", self.support_vector_labels)
        print("Support vector labels length: %s", len(self.support_vector_labels))

    def predict(self, x):
        # instead of calculating W, we calculate (W^T)*x directly because of the kernel function
        prediction = self.bias
        # normalize the test data
        x = preprocessing.scale(x)
        print('bias',self.bias)
        for i in range(len(self.useful_lagrange_multipliers)):
            prediction += self.useful_lagrange_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], x)
        predict_label = int(np.sign(prediction))

        return predict_label





