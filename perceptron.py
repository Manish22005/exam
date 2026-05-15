# Perceptron Learning Algorithm with Decision Region

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

# Input Data
X = np.array([
    [2, 3],
    [1, 1],
    [2, 1],
    [3, 2],
    [6, 5],
    [7, 7],
    [8, 6],
    [9, 8]
])

# Output Labels
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train Perceptron Model
model = Perceptron()

model.fit(X, Y)

# Plot Decision Region
plot_decision_regions(X, Y, clf=model)

# Labels and Title
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Region")

plt.show()