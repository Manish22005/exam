# # Back Propagation Feed Forward Neural Network

# import numpy as np

# # Input Dataset
# X = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])

# # Expected Output
# Y = np.array([
#     [0],
#     [1],
#     [1],
#     [0]
# ])

# # Sigmoid Activation Function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Derivative of Sigmoid
# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Initialize Weights
# np.random.seed(1)

# input_hidden_weights = np.random.uniform(size=(2, 2))
# hidden_output_weights = np.random.uniform(size=(2, 1))

# learning_rate = 0.1

# # Training the Network
# for epoch in range(10000):

#     # Feed Forward
#     hidden_input = np.dot(X, input_hidden_weights)
#     hidden_output = sigmoid(hidden_input)

#     final_input = np.dot(hidden_output, hidden_output_weights)
#     predicted_output = sigmoid(final_input)

#     # Error Calculation
#     error = Y - predicted_output

#     # Back Propagation
#     d_output = error * sigmoid_derivative(predicted_output)

#     error_hidden = d_output.dot(hidden_output_weights.T)

#     d_hidden = error_hidden * sigmoid_derivative(hidden_output)

#     # Update Weights
#     hidden_output_weights += hidden_output.T.dot(d_output) * learning_rate

#     input_hidden_weights += X.T.dot(d_hidden) * learning_rate

# # Final Output
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

lr = 0.1

# Training
for i in range(10000):

    # Feed Forward
    h = sigmoid(np.dot(X, w1))
    o = sigmoid(np.dot(h, w2))

    # Error
    error = Y - o

    # Backpropagation
    d_output = error * derivative(o)

    d_hidden = d_output.dot(w2.T) * derivative(h)

    # Weight Update
    w2 += h.T.dot(d_output) * lr
    w1 += X.T.dot(d_hidden) * lr

# Output
print(o)