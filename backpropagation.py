# Backpropagation Network for XOR Function

# import numpy as np

# # XOR Input and Output
# X = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# Y = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])

# # Sigmoid Function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Sigmoid Derivative
# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Random Weights
# np.random.seed(1)

# weights_input_hidden = np.random.uniform(size=(2, 2))
# weights_hidden_output = np.random.uniform(size=(2, 1))

# learning_rate = 0.1

# # Training
# for epoch in range(10000):

#     hidden_input = np.dot(X, weights_input_hidden)
#     hidden_output = sigmoid(hidden_input)

#     final_input = np.dot(hidden_output, weights_hidden_output)
#     predicted_output = sigmoid(final_input)

#     error = Y - predicted_output

#     d_predicted_output = error * sigmoid_derivative(predicted_output)

#     error_hidden = d_predicted_output.dot(weights_hidden_output.T)

#     d_hidden_layer = error_hidden * sigmoid_derivative(hidden_output)

#     weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate

#     weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# # Output
# print("Predicted Output:\n")
# print(predicted_output)


import numpy as np

# XOR Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def derivative(x):
    return x * (1 - x)

# Weights
np.random.seed(1)
w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

# Training
for i in range(10000):

    h = sigmoid(np.dot(X, w1))
    o = sigmoid(np.dot(h, w2))

    error = Y - o

    d_output = error * derivative(o)
    d_hidden = d_output.dot(w2.T) * derivative(h)

    w2 += h.T.dot(d_output) * 0.1
    w1 += X.T.dot(d_hidden) * 0.1

# Output
print(o)