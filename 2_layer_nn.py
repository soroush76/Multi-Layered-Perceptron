import numpy as np
import csv
from sklearn.model_selection import train_test_split

def load_data(): # load dataset from breast cancer file
    samples = []
    labels = []

    with open('breastcancer_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1] == 'M': labels.append(1)
            else: labels.append(0)

            samples.append([float(i) for i in row[2:]])

    samples, labels = np.array(samples), np.array(labels).reshape(-1, 1) # prevent returning a ranked-1-array
    samples = (samples - np.mean(samples, axis=1).reshape(-1, 1))/np.std(samples, axis=1).reshape(-1, 1) # normalize dataset

    return samples, labels

def initialize_parameters(n_input, n_hidden): # initialize parameters with tiny random numbers
    w1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1)) #np.random.randn(n_hidden, 1)
    w2 = np.random.randn(1, n_hidden) * 0.01
    b2 = np.zeros((1, 1)) #np.random.randn(1, 1)

    return {'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2}

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    z_temp = z.copy()
    z_temp[z < 0] = 0
    return z_temp

def relu_derivative(z):
    z_temp = np.zeros((z.shape[0], z.shape[1]))
    z_temp[z >= 0] = 1
    return z_temp

def forward_prop(x, parameters, activation):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x.T) + b1
    if activation == 'relu': a1 = relu(z1)
    elif activation == 'sigmoid': a1 = sigmoid(z1)
    elif activation == 'tanh': a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

def back_prop(x, y, cache, parameters, activation):
    z1 = cache['z1']
    a1 = cache['a1']
    z2 = cache['z2']
    a2 = cache['a2']

    m = x.shape[0]
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dz2 = a2 - y.T
    dw2 = 1/m * np.dot(dz2, a1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    if activation == 'sigmoid': dz1 = np.dot(dz2.T, w2) * (sigmoid(z1.T)*(1 - sigmoid(z1.T)))
    elif activation == 'relu': dz1 = np.dot(dz2.T, w2) * relu_derivative(z1.T)
    elif activation == 'tanh': dz1 = np.dot(dz2.T, w2) * (1 - np.power(np.tanh(z1.T), 2))
    dw1 = 1/m * np.dot(dz1.T, x)
    db1 = 1/m * np.sum(dz1.T, axis=1, keepdims=True)

    assert(dw1.shape == w1.shape)
    assert(db1.shape == b1.shape)
    assert(dw2.shape == w2.shape)
    assert(db2.shape == b2.shape)

    return {'dz2':dz2, 'dw2':dw2, 'db2':db2, 'dz1':dz1, 'dw1':dw1, 'db1':db1}

def two_Layered_NN(samples, labels, activation, n_hidden, num_iterations, learning_rate, print_cost=False):
    m = samples.shape[0]
    print(samples.shape)
    params = initialize_parameters(samples.shape[1], n_hidden)
    cost_history = []

    for i in range(num_iterations):
        cache = forward_prop(samples, params, activation)

        cost = -1/m * (np.sum(np.log2(cache['a2'])*labels) + np.sum(np.log2(1-cache['a2'])*(1-labels)))
        cost_history.append(cost)

        gradients = back_prop(samples, labels, cache, params, activation)
        params['w1'] = params['w1'] - learning_rate * gradients['dw1']
        params['b1'] = params['b1'] - learning_rate * gradients['db1']
        params['w2'] = params['w2'] - learning_rate * gradients['dw2']
        params['b2'] = params['b2'] - learning_rate * gradients['db2']

        if print_cost and i%1000 == 0: print('cost after iteration {}: {}'.format(i, cost))

    return {'cache': cache, 'parameters':params, 'gradients': gradients, 'cost_history':cost_history}

def predict(x, parameters, activation):
    cache = forward_prop(x, parameters, activation)
    y_hat = cache['a2']
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    return y_hat.T

samples, labels = load_data()
train_data, test_data, train_label, test_label = train_test_split(samples, labels, test_size=0.25, random_state=4)

model = two_Layered_NN(train_data, train_label, activation='relu', n_hidden=50, num_iterations=10000, learning_rate=0.01, print_cost=True)
pred_labels = predict(train_data, parameters=model['parameters'], activation='relu')
print('accuracy on train set:', (np.sum(pred_labels == train_label)/train_label.size) * 100)
pred_labels = predict(test_data, parameters=model['parameters'], activation='relu')
print('accuracy on test set:', (np.sum(pred_labels == test_label)/test_label.size) * 100)
