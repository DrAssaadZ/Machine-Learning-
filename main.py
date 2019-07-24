from __future__ import print_function
import matplotlib, sys

from matplotlib import pyplot as plt
import numpy as np


def plot(matrix, weights=None, title="Prediction Matrix"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    if weights != None:
        map_min = 0.0
        map_max = 1.1
        y_res = 0.001
        x_res = 0.001
        ys = np.arange(map_min, map_max, y_res)
        xs = np.arange(map_min, map_max, x_res)
        zs = []
        for cur_y in np.arange(map_min, map_max, y_res):
            for cur_x in np.arange(map_min, map_max, x_res):
                zs.append(predict([1.0, cur_x, cur_y], weights))
        xs, ys = np.meshgrid(xs, ys)
        zs = np.array(zs)
        zs = zs.reshape(xs.shape)
        cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b', 'r'), alpha=0.1)

    c1_data = [[], []]
    c0_data = [[], []]
    for i in range(len(matrix)):
        cur_i1 = matrix[i][1]
        cur_i2 = matrix[i][2]
        cur_y = matrix[i][-1]
        if cur_y == 1:
            c1_data[0].append(cur_i1)
            c1_data[1].append(cur_i2)
        else:
            c0_data[0].append(cur_i1)
            c0_data[1].append(cur_i2)

    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label='Class -1')
    c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label='Class 1')

    plt.legend(fontsize=10, loc=1)
    plt.show()
    return

    print("Matrix dimensions not covered.")


def predict(inputs, weights):
    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i * w
    return 1.0 if activation >= 0.0 else 0.0


# each matrix row: up to last row = inputs, last row = y (classification)
def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:3], weights)  # get predicted classification
        preds.append(pred)
        if pred == matrix[i][-1]: num_correct += 1.0
    print("Predictions:", preds)
    return num_correct / float(len(matrix))


# each matrix row: up to last row = inputs, last row = y (classification)
def train_weights(matrix, weights, nb_epoch=10, l_rate=1.00, do_plot=False, stop_early=True, verbose=False):
    for epoch in range(nb_epoch):
        cur_acc = accuracy(matrix, weights)
        print("\nEpoch %d \nWeights: " % epoch, weights)
        print("Accuracy: ", cur_acc)

        if cur_acc == 1.0 and stop_early:
            break
        # if do_plot and len(matrix[0])==4: plot(matrix,weights) # if 2D inputs, excluding bias
        if do_plot:
            plot(matrix, weights, title="Epoch %d" % epoch)

        for i in range(len(matrix)):
            prediction = predict(matrix[i][:3], weights)  # get predicted classification
            error = matrix[i][3] - prediction  # get error from real classification
            # if verbose: sys.stdout.write("Training on data at index %d...\n" % (i))
            for j in range(len(weights)):  # calculate new weight for each node
                # if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> " % (j, weights[j]))
                weights[j] = weights[j] + (l_rate * error * matrix[i][j])
                # if verbose: sys.stdout.write("%0.5f\n" % (weights[j]))

    # if len(matrix[0])==4: plot(matrix,weights) # if 2D inputs, excluding bias
    plot(matrix, weights, title="Final Epoch")
    return weights


def exemple():
    nb_epoch = 1000
    alpha = 1.0
    plot_each_epoch = False
    stop_early = True

                   #  x0    x1    x2    y
    exemple_base = [[1.00, 0.08, 0.72, 1.0],
                    [1.00, 0.10, 1.00, 0.0],
                    [1.00, 0.26, 0.58, 1.0],
                    [1.00, 0.35, 0.95, 0.0],
                    [1.00, 0.45, 0.15, 1.0],
                    [1.00, 0.60, 0.30, 1.0],
                    [1.00, 0.70, 0.65, 0.0],
                    [1.00, 0.70, 0.45, 0.0],
                    [1.00, 0.92, 0.45, 0.0]]
    weights = [-0.60, 0.7, -1.15]  # initial weights specified in problem[	0.20,	1.00,  -1.00]

    train_weights(exemple_base, weights=weights, nb_epoch=nb_epoch, l_rate=alpha, do_plot=plot_each_epoch
                  , stop_early=stop_early)


exemple()