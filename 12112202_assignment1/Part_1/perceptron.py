import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1000, learning_rate=0.001):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate  # Fill in: Initialize learning rate
        self.weights = np.zeros(len(n_inputs[0]))
        self.bias = 0

        # Fill in: Initialize weights with zeros


    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): Input array of training data, input vec must be all samples
        Returns:
            int: Predicted label (1 or -1) or Predicted lables.
        """
        return np.sign(np.dot(input_vec, self.weights) + self.bias)


    def train(self, training_inputs, labels):
        """
        Trains the perceptron.
        Args:
            training_inputs (list of np.ndarray): List of numpy arrays of training points.
            labels (np.ndarray): Array of expected output values for the corresponding point in training_inputs.
        """
        # we need max_epochs to train our model
        for _ in range(self.max_epochs):
            """
                What we should do in one epoch ? 
                you are required to write code for 
                1.do forward pass
                2.calculate the error
                3.compute parameters' gradient 
                4.Using gradient descent method to update parameters(not Stochastic gradient descent!,
                please follow the algorithm procedure in "perceptron_tutorial.pdf".)
            """
            miss = 0
            weights_grad = np.zeros(len(training_inputs[0]))
            bias_grad = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.forward(inputs)
                if prediction != label:
                    miss += 1
                    weights_grad += label * inputs
                    bias_grad += label


            if miss == 0:
                break
            self.bias += bias_grad/miss
            self.weights += weights_grad/miss

        return self.weights, self.bias


if __name__ == "__main__":
    mean1 = [2, 3]
    covariance1 = [[1, 0.5], [0.5, 1]]
    mean2 = [-2, -3]
    covariance2 = [[1, -0.5], [-0.5, 1]]

    points1_train = np.random.multivariate_normal(mean1, covariance1, 80)
    points1_test = np.random.multivariate_normal(mean1, covariance1, 20)
    points2_train = np.random.multivariate_normal(mean2, covariance2, 80)
    points2_test = np.random.multivariate_normal(mean2, covariance2, 20)

    training_inputs = np.concatenate((points1_train, points2_train), axis=0)
    testing_inputs = np.concatenate((points1_test, points2_test), axis=0)
    labels = np.concatenate((np.ones(80), -np.ones(80)))
    test_labels = np.concatenate((np.ones(20), -np.ones(20)))

    training_data = np.column_stack((training_inputs, labels))
    testing_data = np.column_stack((testing_inputs, test_labels))





    Perceptron = Perceptron(training_inputs)
    weights, bias = Perceptron.train(training_inputs, labels)
    print("accurary:", np.mean(Perceptron.forward(testing_inputs) == test_labels))

    plt.figure()
    plt.scatter(points1_train[:, 0], points1_train[:, 1], c='r', label='class 1_train')
    plt.scatter(points2_train[:, 0], points2_train[:, 1], c='b', label='class 2_train')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.show()
    plt.figure()
    plt.scatter(points1_test[:, 0], points1_test[:, 1], c='r', label='class 1_test')
    plt.scatter(points2_test[:, 0], points2_test[:, 1], c='b', label='class 2_test')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')

    plt.legend()
    plt.show()


    # mean too close
    mean1 = [2, 3]
    covariance1 = [[1, 0.5], [0.5, 1]]
    mean2 = [2.5, 3.5]  # Modified mean closer to mean1
    covariance2 = [[1, -0.5], [-0.5, 1]]

    points1_train = np.random.multivariate_normal(mean1, covariance1, 80)
    points1_test = np.random.multivariate_normal(mean1, covariance1, 20)
    points2_train = np.random.multivariate_normal(mean2, covariance2, 80)
    points2_test = np.random.multivariate_normal(mean2, covariance2, 20)

    training_inputs = np.concatenate((points1_train, points2_train), axis=0)
    testing_inputs = np.concatenate((points1_test, points2_test), axis=0)
    labels = np.concatenate((np.ones(80), -np.ones(80)))
    test_labels = np.concatenate((np.ones(20), -np.ones(20)))

    training_data = np.column_stack((training_inputs, labels))
    testing_data = np.column_stack((testing_inputs, test_labels))





    Perceptron = Perceptron
    weights, bias = Perceptron.train(training_inputs, labels)
    print("accurary:", np.mean(Perceptron.forward(testing_inputs) == test_labels))

    plt.figure()
    plt.scatter(points1_train[:, 0], points1_train[:, 1], c='r', label='class 1_train')
    plt.scatter(points2_train[:, 0], points2_train[:, 1], c='b', label='class 2_train')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.show()
    plt.figure()
    plt.scatter(points1_test[:, 0], points1_test[:, 1], c='r', label='class 1_test')
    plt.scatter(points2_test[:, 0], points2_test[:, 1], c='b', label='class 2_test')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')

    plt.legend()
    plt.show()
    # variance is too high
    mean1 = [2, 3]
    # Increase variance for class 1
    #accurary: 0.925 for [[4, 2], [2, 4]]
    #accurary: 0.9 for [[10, 5], [5, 10]]
    #accurary: 0.225 for [[50, 25], [25, 50]]

    covariance1 = [[50, 25], [25, 50]]
    mean2 = [-2, -3]
    # Increase variance for class 2
    covariance2 = [[50, -25], [-25, 50]]
    points1_train = np.random.multivariate_normal(mean1, covariance1, 80)
    points1_test = np.random.multivariate_normal(mean1, covariance1, 20)
    points2_train = np.random.multivariate_normal(mean2, covariance2, 80)
    points2_test = np.random.multivariate_normal(mean2, covariance2, 20)

    training_inputs = np.concatenate((points1_train, points2_train), axis=0)
    testing_inputs = np.concatenate((points1_test, points2_test), axis=0)
    labels = np.concatenate((np.ones(80), -np.ones(80)))
    test_labels = np.concatenate((np.ones(20), -np.ones(20)))

    training_data = np.column_stack((training_inputs, labels))
    testing_data = np.column_stack((testing_inputs, test_labels))





    Perceptron = Perceptron
    weights, bias = Perceptron.train(training_inputs, labels)
    print("accurary:", np.mean(Perceptron.forward(testing_inputs) == test_labels))

    plt.figure()
    plt.scatter(points1_train[:, 0], points1_train[:, 1], c='r', label='class 1_train')
    plt.scatter(points2_train[:, 0], points2_train[:, 1], c='b', label='class 2_train')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.show()
    plt.figure()
    plt.scatter(points1_test[:, 0], points1_test[:, 1], c='r', label='class 1_test')
    plt.scatter(points2_test[:, 0], points2_test[:, 1], c='b', label='class 2_test')
    # plot the decision boundary
    x = np.linspace(-6, 6, 100)
    y = - (weights[0] * x + bias) / weights[1]
    plt.plot(x, y, label='Decision Boundary')

    plt.legend()
    plt.show()