import numpy as np
import random as rnd
from matplotlib import pyplot as plt


# function that fills the dataset's 1st and 2nd column with random numbers between 1 and 0 and returns the result matrix
def fill_dataset(mat):
    for i in range(len(mat)):
        mat[i][0] = round(rnd.uniform(0, 1), 3)
        mat[i][1] = round(rnd.uniform(0, 1), 2)
    return mat


# function that calculates the dataset output y either 1 or -1
def output_dataset(mat):
    for i in range(len(mat)):
        if mat[i][0] + mat[i][1] - 1 > 0:
            mat[i][2] = 1
        else:
            mat[i][2] = -1
    return mat


# function that initialises the neuron ( bias = 0.5 , w1, w2 = random number between 0 and 1
def init_neuron(neuron):
    neuron[0] = 0.5
    neuron[1] = round(rnd.uniform(0, 1), 3)
    neuron[2] = round(rnd.uniform(0, 1), 3)
    return neuron


# function that calculates the neuron's output (activation method) either 1 or 0
def output_neuron(neuron, data, row):
    # use dot function
    activation = neuron[1] * data[row, 0] + neuron[2] * data[row, 1] + neuron[0]
    if activation > 0:
        neuron[3] = 1
    else:
        neuron[3] = -1
    return activation


# function that updates the neuron's bias and weights
def update_neuron(neuron, data, row):
    neuron[0] = neuron[0] + alpha*(data[row, 2] - output_neuron(neuron, data, row))
    neuron[1] = neuron[1] + alpha*(data[row, 2] - output_neuron(neuron, data, row)) * data[row, 0]
    neuron[2] = neuron[2] + alpha*(data[row, 2] - output_neuron(neuron, data, row)) * data[row, 1]


# Main program
# initialising the dataset, neuron , learning rate ( alpha) and the errors array
dataset = output_dataset(fill_dataset(np.zeros((50, 3))))
print(dataset)
neuron = init_neuron(np.zeros((4, 1)))
alpha = 0.02
errors = []

for epoch in range(1000):
    nbr_error = 0
    for j in range(len(dataset)):

        if neuron[3] != dataset[j, 2]:
            nbr_error = (dataset[j, 2] - neuron[3])*(dataset[j, 2] - neuron[3])
            update_neuron(neuron, dataset, j)
    print("epock number :", epoch)
    print("number of errors is : ", nbr_error)
    print("neuron is :\n", neuron)
    errors.append((nbr_error))
    if nbr_error == 0:
        break

# plotting the number of errors curve
plt.title("Number of errors evolution per epoch")
plt.ylabel("Number of errors")
plt.xlabel("Number of epoches")
plt.plot(errors)
plt.show()


# plotting the data
for i in range(len(dataset)):
    if dataset[i, 2] == 1:
        color = 'b'
    else:
        color = 'r'
    plt.scatter(dataset[i, 0], dataset[i, 1], s=40.0, c=color, label='Class 1')

# plot settings
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
x = np.linspace(-5, 5, 100)

# plotting the straight line that divides the two classes
plt.plot(x, (neuron[1]/neuron[2]) * -1 * x + (neuron[0]/neuron[2]) * -1, ':g')
plt.show()