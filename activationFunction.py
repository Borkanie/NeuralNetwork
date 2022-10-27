import math
import matplotlib.pyplot as plt
import numpy

def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
    if x>0:
        return x
    else:
        return 0

def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

if __name__ == "__main__":
    resultSigmoid=[]
    resultRelu=[]
    resultTanh=[]
    for x in numpy.arange(-10,10,0.1):
        resultSigmoid.append([x,sigmoid(x)])
        resultRelu.append([x,relu(x)])
        resultTanh.append([x,tanh(x)])
    # what does zip do?
    # well 'The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and returns it.'
    fig, axs = plt.subplots(3)
    fig.suptitle('Activation functions')
    axs[0].plot(*zip(*resultSigmoid))
    axs[0].text(0,0,"Sigmoid")
    axs[1].plot(*zip(*resultRelu))
    axs[1].text(0,0,"Relu")
    axs[2].plot(*zip(*resultTanh))
    axs[2].text(0,0,"Tanh")
    plt.show()
