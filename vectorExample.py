import numpy as np
from activationFunction import sigmoid

def makePrediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector,weights,bias)
    layer_2 = sigmoid(layer_1)
    return layer_2

def secondDemo():
    input_vector = np.array([1.66, 1.56])
    weights_1 = np.array([1.45, -0.66])
    bias = np.array([0.0])
    prediction = makePrediction(input_vector,weights_1,bias)
    print(f"The prediction result is: {prediction}")


def firstDemo():
    input_vector = [1.72, 1.23]
    weights_1 = [1.26, 0]
    weights_2 = [2.17, 0.32]

    # Computing the dot product of input_vector and weights_1
    dot_product_1 = np.dot(input_vector, weights_1)
    print(f"The first dot product is:{dot_product_1}")

    # Computing the dot product of input_vector and weights_2
    dot_product_2 = np.dot(input_vector, weights_2)
    print(f"The first dot product is:{dot_product_2}")


if __name__ == "__main__":
    secondDemo()
