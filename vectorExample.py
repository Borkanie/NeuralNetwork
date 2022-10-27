import numpy as np
from NerualNetwork import NeuralNetwork
from activationFunction import sigmoid
import matplotlib.pyplot as plt


def makePrediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights)
    layer_1 = layer_1 + bias
    layer_2 = sigmoid(layer_1)
    return layer_2


def getResult(prediction):
    if prediction > 0.5:
        return True
    else:
        return False


def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))


def secondDemo():
    # ON THE FIRST PREDICITON THE RESULT SHOULD BE 1
    input_vector = np.array([1.66, 1.56])
    weights_1 = np.array([1.45, -0.66])
    bias = np.array([0.0])
    prediction = makePrediction(input_vector, weights_1, bias)
    print(f"The FIRST prediction result is: {prediction}")
    target = 1
    mse = np.square(target - prediction)
    print(f"Error on the first prediction is:{mse}")
    # ON THE SECOND PREDICTION THE RESULT SHOULD BE 0
    input_vector = np.array([2, 1.5])
    prediction = makePrediction(input_vector, weights_1, bias)
    print(f"The SECOND prediction result is: {prediction}")
    # THE RESULT IS 1 SO WE COMPUTE ERROR NOW
    # WE USE MEAN SQUARE ERROR AS A COST FUNCTION TO ASSES ERROR
    target = 0
    mse = np.square(target - prediction)
    print(f"Error on the SECOND prediction is:{mse}")
    gradient = 2*(target - prediction)
    print(f"Gradient  of SECOND prediciton is:{gradient}")
    # SINCE THE ERROR FUNCTION IS A SQUARED FUNCTION IT'S GRAPHIC IS A PARABOLA
    # SINCE WE HAVE NO FIRST ORDER MEMBER IN THE ERROR FUNCTION IT IS CENTERD IN 0
    # THAT MEANS THAT IF THE GRADIENT OF THE COST FUNCTION IS POZITIVE WE WANT TO DECREASE THE VALUE OF TH WIGHTS
    # AND IF THE GRADIENT IS NEGATIVE THAN WE WANT TO INCREASE THE WIEGHTS

    # THE PREDICITON CAN JUMP FROM ONE EXTREME TO THE OTHER SO WE ADD A SLOWING FACTOR
    alpha = 0.1
    # THIS IS CALLED LEARNING RATE

    if gradient > 0:
        weights_1 = weights_1 - alpha*gradient
    else:
        weights_1 = weights_1 + alpha*gradient

    # HERE WE MAKE A NEW PREDICTION
    prediction = makePrediction(input_vector, weights_1, bias)
    print(f"The SECOND prediction result is: {prediction}")
    # THE RESULT IS 1 SO WE COMPUTE ERROR NOW
    # WE USE MEAN SQUARE ERROR AS A COST FUNCTION TO ASSES ERROR
    target = 0
    mse = np.square(target - prediction)
    print(f"Error on the SECOND prediction is:{mse}")


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


def thirdDemo():
    input_vectors = np.array(
        [
            [3, 1.5],
            [2, 1],
            [4, 1.5],
            [3, 4],
            [3.5, 0.5],
            [2, 0.5],
            [5.5, 1],
            [1, 1],
        ]
    )
    targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    learning_rate = 0.1
    neural_network = NeuralNetwork(learning_rate)
    training_error = neural_network.train(input_vectors, targets, 10000)
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.show()

if __name__ == "__main__":
    thirdDemo()
