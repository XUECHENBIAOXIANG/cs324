import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from mlp_numpy import MLP  
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
num_samples = 1000
num_features = 10
num_classes = 2

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(pred_classes == true_classes)



def generate_data():

    X = np.random.randn(num_samples, num_features)  # Features
    y = np.random.randint(0, num_classes, size=num_samples)  # Labels (multi-class classification)
    # Perform one-hot encoding on labels
    one_hot_encoder = OneHotEncoder(categories='auto')
    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1)).toarray()

    return X, y_one_hot


def train_and_test_data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, batch=True, batch_size=1):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    # X, y = generate_data()
    finished = False
    time = -1
    X, y = make_moons(n_samples=num_samples, noise=0.1, random_state=42)
    one_hot_encoder = OneHotEncoder(categories='auto')
    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1)).toarray()
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_and_test_data_split(X, y_one_hot)

    if not batch:
        if batch_size > X_train.shape[0]:
            assert batch_size > X_train.shape[0], "Batch size cannot be larger than the training set size."


    MLP_model = MLP(X_train.shape[1], [int(i) for i in dnn_hidden_units.split(',')], 2, learning_rate)
    accuracy_list = []
    accuracy_list_test = []
    loss_list=[]
    loss_list_test=[]
    if batch:
        batch_size = X_train.shape[0]
    for step in range(max_steps):
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]



        # TODO: Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Forward pass
            res = MLP_model.forward(X_batch)
            softmax = MLP_model.SoftMax.forward(res)
            loss = MLP_model.CrossEntropy.forward(softmax, y_batch)

            # Backward pass
            back_cross = MLP_model.CrossEntropy.backward(softmax, y_batch)
            MLP_model.backward(back_cross)

        test_res = MLP_model.testforward(X_test)
        test_softmax = MLP_model.SoftMax.forward(test_res)
        test_loss = MLP_model.CrossEntropy.forward(test_softmax, y_test)
        test_accuracy = accuracy(test_res, y_test)
        accuracy_list_test.append(test_accuracy)
        loss_list_test.append(test_loss)


        train_res = MLP_model.testforward(X_train)
        train_softmax = MLP_model.SoftMax.forward(train_res)
        train_loss = MLP_model.CrossEntropy.forward(train_softmax,y_train)
        train_accuracy = accuracy(train_res, y_train)
        accuracy_list.append(train_accuracy)
        loss_list.append(train_loss)
        if test_accuracy == 1:
            if not finished:
                time = step
                finished = True
        if step % eval_freq == 0 or step == max_steps - 1:
            # TODO: Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy

            # print(f"Step: {step}, Loss: {loss}, Accuracy: {accuracy(res, y_train)}")
            print(f"Step: {step}, Loss: {test_loss}, Accuracy: {test_accuracy}")

    print("Training complete!")
    return X, y,accuracy_list,accuracy_list_test,time,loss_list,loss_list_test

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    #accept a parameter that allows the user to specify if the training has to be performed using batch gradient descent
    parser.add_argument('--batch', type=bool, default=True,
                        help='Batch Gradient Descent')
    parser.add_argument('--batch_size', type=int, default=1,help='Batch size')

    FLAGS = parser.parse_known_args()[0]
    
    return train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
