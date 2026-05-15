# # ==========================================
# # ACTIVATION FUNCTIONS IN NEURAL NETWORKS
# # Combined Program with Graphs
# # ==========================================

# # Import Libraries
# import math
# import numpy as np
# import matplotlib.pyplot as plt


# # ==========================================
# # ACTIVATION FUNCTIONS
# # ==========================================

# # 1. Linear Activation Function
# def linear(x):
#     return x


# # 2. Sigmoid Activation Function
# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))


# # 3. Tanh Activation Function
# def tanh(x):
#     return math.tanh(x)


# # 4. ReLU Activation Function
# def relu(x):
#     return max(0, x)


# # 5. Step Activation Function
# def step(x):
#     if x >= 0:
#         return 1
#     else:
#         return 0


# # 6. Leaky ReLU Activation Function
# def leaky_relu(x):
#     if x > 0:
#         return x
#     else:
#         return 0.01 * x


# # ==========================================
# # TESTING WITH SAMPLE INPUTS
# # ==========================================

# inputs = [-10, -5, 0, 5, 10]

# print("-------------------------------------------------------------")
# print("Input | Linear | Sigmoid | Tanh | ReLU | Step | Leaky ReLU")
# print("-------------------------------------------------------------")

# for x in inputs:
    
#     linear_output = linear(x)
#     sigmoid_output = round(sigmoid(x), 4)
#     tanh_output = round(tanh(x), 4)
#     relu_output = relu(x)
#     step_output = step(x)
#     leaky_output = round(leaky_relu(x), 4)

#     print(f"{x:>5} | {linear_output:>6} | {sigmoid_output:>7} | {tanh_output:>5} | {relu_output:>4} | {step_output:>4} | {leaky_output:>10}")


# # ==========================================
# # GRAPH PLOTTING
# # ==========================================

# # Generate Input Values
# x = np.linspace(-10, 10, 200)

# # Generate Outputs
# y_linear = x

# y_sigmoid = 1 / (1 + np.exp(-x))

# y_tanh = np.tanh(x)

# y_relu = np.maximum(0, x)

# y_step = np.where(x >= 0, 1, 0)

# y_leaky = np.where(x > 0, x, 0.01 * x)


# # ==========================================
# # PLOT ALL GRAPHS
# # ==========================================

# plt.figure(figsize=(10, 8))

# # Linear
# plt.plot(x, y_linear, label="Linear")

# # Sigmoid
# plt.plot(x, y_sigmoid, label="Sigmoid")

# # Tanh
# plt.plot(x, y_tanh, label="Tanh")

# # ReLU
# plt.plot(x, y_relu, label="ReLU")

# # Step
# plt.plot(x, y_step, label="Step")

# # Leaky ReLU
# plt.plot(x, y_leaky, label="Leaky ReLU")


# # ==========================================
# # GRAPH SETTINGS
# # ==========================================

# plt.xlabel("Input")
# plt.ylabel("Output")

# plt.title("Activation Functions in Neural Networks")

# plt.legend()

# plt.grid(True)

# plt.show()




import math
import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

def relu(x):
    return max(0, x)

def step(x):
    return 1 if x >= 0 else 0

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

# Sample Inputs
inputs = [-10, -5, 0, 5, 10]

print("Input | Linear | Sigmoid | Tanh | ReLU | Step | Leaky")
print("-" * 55)

for x in inputs:
    print(
        x,
        "|",
        linear(x),
        "|",
        round(sigmoid(x), 4),
        "|",
        round(tanh(x), 4),
        "|",
        relu(x),
        "|",
        step(x),
        "|",
        round(leaky_relu(x), 4)
    )

# Graph Data
x = np.linspace(-10, 10, 200)

plt.plot(x, x, label="Linear")
plt.plot(x, 1 / (1 + np.exp(-x)), label="Sigmoid")
plt.plot(x, np.tanh(x), label="Tanh")
plt.plot(x, np.maximum(0, x), label="ReLU")
plt.plot(x, np.where(x >= 0, 1, 0), label="Step")
plt.plot(x, np.where(x > 0, x, 0.01*x), label="Leaky ReLU")

# Graph Settings
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Activation Functions")
plt.legend()
plt.grid(True)

plt.show()