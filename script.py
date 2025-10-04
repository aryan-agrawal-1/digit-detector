import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def init_params():
    # Used to be 10 neurons - increased to 128 for better performance

    W1 = np.random.randn(128, 784) * np.sqrt(2/784) # He initialization
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.randn(10, 128) * np.sqrt(2/128) # He initialization
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLu(Z):
    return np.maximum(Z, 0) # This goes through every elt in Z and returns 0 if lt 0 otherwise will just give the elt back

def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

# Encode the labels as arrays
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Creating an m x 10 array of zeroes
    one_hot_Y[np.arange(Y.size), Y] = 1 # For each row, go to column Y and set it to 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLu(Z):
    return Z > 0 # will return 1 if true and 0 if false

def back_prop(Z1, A1, Z2, A2, W2, Y, X):
    m = X.shape[1]  # Number of training examples
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)

    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def save_model(W1, b1, W2, b2, filename='model_weights.npz'):
    # Save all the weights and biases to a file so we don't have to retrain every time
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Model saved to {filename}")

def load_model(filename='model_weights.npz'):
    # Load the weights and biases from a file
    data = np.load(filename)
    return data['W1'], data['b1'], data['W2'], data['b2']

def get_predictions(A):
    # Get the index of the highest probability (our prediction)
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    print(predictions)
    print(Y)
    return np.sum(predictions == Y) / Y.size

# Running the loop
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    # How many times we want to run the loop
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, Y, X)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0:
            print(f"Iteration: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2

# Testing specific images
def make_predictions(X, W1, b1, W2, b2):
    _,_,_,A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_predictions(index, W1, b1, W2, b2):
    # Get the column of values of the current image
    current_image = X_train[:, index, None]

    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print("Our prediction: ", prediction)
    print("Real answer: ", label)

    current_image = current_image.reshape((28, 28)) * 255 # Make it a 28 x 28 array of the values from 0-255
    
    # Set colourmap to gray and create the image and then show it
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    # Creating pandas dataframe from csv
    data = pd.read_csv('./datasets/mnist_train.csv')

    # Converting to np array so we can do cool maths

    # We are setting aside some data to ensure we dont overfit, this will be used for cross-validation

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T # Transpose the matrix
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255. # This is now a (784, m_train) array with m columns of images, the values are now all between 0 and 1
    _,m_train = X_train.shape

    # Train the model
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.3)

    # Save the trained model so we can use it later without retraining
    save_model(W1, b1, W2, b2)

    # Cross validation (test on untrained images)
    dev_predictions = make_predictions(X_dev,  W1, b1, W2, b2)
    print(get_accuracy(dev_predictions, Y_dev))

