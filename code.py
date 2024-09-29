# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Load the dataset
dataset = pd.read_csv('C:\MY_WORKSPACE\diet\AI\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # Using Annual Income and Spending Score

# Initializing the SOM
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the SOM
plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plotting the distance map as a heatmap
plt.colorbar()

# Plotting markers for each customer
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)  # Getting the winning node
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[0], markerfacecolor='None',
             markeredgecolor=colors[0], markersize=10, markeredgewidth=2)

plt.title('Customer Segments using Self-Organizing Map')
plt.show()
