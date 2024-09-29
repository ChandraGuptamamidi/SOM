1. Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

------
pandas (pd): A powerful library for data manipulation and analysis. It is used here to read the CSV file and preprocess the data.
numpy (np): Used for handling numerical operations, such as array manipulations.
matplotlib.pyplot (plt): A library used for data visualization. Here, it's used to create visual representations of the SOM and clusters.
minisom: A third-party library specifically used for creating Self-Organizing Maps. MiniSom is a class that helps implement the SOM.
------
2. Loading and Preprocessing the Dataset

dataset = pd.read_csv('C:\MY_WORKSPACE\diet\AI\Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # Using Annual Income and Spending Score

pd.read_csv('data/Mall_Customers.csv'): Reads the dataset file (Mall_Customers.csv) located in the data folder.
dataset.iloc[:, [3, 4]]: Selects specific columns from the DataFrame. The .iloc[] method allows selection based on index positions:
Column 3 (Annual Income) and column 4 (Spending Score) are chosen for clustering.
.values: Converts the selected DataFrame into a NumPy array (X). This array will be fed into the SOM for training.


3. Initializing the SOM

som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)
--------
This line initializes a 2D Self-Organizing Map grid using the MiniSom class. Break down each parameter:

x=10, y=10: Specifies the size of the SOM grid (10x10 nodes).

Larger Grid Size: More granularity in clustering. Useful for complex datasets.
Smaller Grid Size: Fewer nodes and coarser clustering, suitable for small datasets.
Choosing x and y: Typically, the size is proportional to the square root of the number of input vectors. For larger datasets, a bigger grid may be used.
input_len=2: Refers to the number of features in the input vector.

Our input vector has only two features: Annual Income and Spending Score.
sigma=1.0: The spread of the neighborhood function.

This controls how neighboring nodes in the grid are influenced during weight updates.
Larger sigma values: More nodes are influenced (smooth updating).
Smaller sigma values: Fewer nodes are influenced (more local updating).
Default: Typically between 0.1 and 2.0. The value should decrease over time to fine-tune specific areas.
learning_rate=0.5: Determines the rate at which the SOM learns.

Controls the step size during each update of the node’s weight.
Larger learning rates: Faster learning initially but risk of overshooting.
Smaller learning rates: Slower convergence, but more stable learning.
Typical Range: Between 0.1 and 0.5, and it should decrease over time.
------

4. Randomly Initializing Weights

som.random_weights_init(X)

som.random_weights_init(X): Initializes the weight vectors for each node randomly based on the distribution of the input data (X).
Random weight initialization helps the SOM start with diverse weight vectors, which improves learning.
This method ensures that each node's initial weights are unique and based on the actual range of the data.

5. Training the SOM

som.train_random(data=X, num_iteration=100)
som.train_random(...): Trains the SOM using the input data X for a specified number of iterations (num_iteration).

Parameters:

data=X: The input data for training.
num_iteration=100: Number of times the SOM will pass through the dataset.
Larger values: More iterations result in better convergence.
Smaller values: May not converge properly.
For a dataset with a few thousand entries, around 100 to 500 iterations are usually enough.

6. Visualizing the SOM Distance Map


plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plotting the distance map as a heatmap
plt.colorbar()

-------
plt.figure(figsize=(10, 7)): Creates a new figure with a specific size.

figsize=(width, height) sets the width and height of the figure in inches.
som.distance_map(): Calculates the mean inter-neuron distance for each node.

Returns a 2D array where each element represents the distance of that node to its neighboring nodes.
This distance map helps visualize cluster boundaries.
plt.pcolor(som.distance_map().T, cmap='coolwarm'): Creates a heatmap representation of the SOM distance map.

T: Transposes the matrix so that the map is oriented correctly.
cmap='coolwarm': Specifies the colormap. Nodes with high values (red) are farther from their neighbors, indicating possible cluster boundaries.
plt.colorbar(): Adds a color bar next to the heatmap to interpret the scale of distances visually.
---------

7. Plotting Markers for Each Data Point

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)  # Getting the winning node
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[0], markerfacecolor='None',
             markeredgecolor=colors[0], markersize=10, markeredgewidth=2)
Markers and Colors:
------
markers = ['o', 's'] defines shapes to distinguish clusters (e.g., 'o' is a circle, 's' is a square).
colors = ['r', 'g'] specifies colors for different clusters.
for i, x in enumerate(X):

Loops through each data point x in the input array X.
i is the index, and x is the data point itself.
w = som.winner(x):

Finds the Best Matching Unit (BMU) for each data point x.
The winner() method returns the (x, y) position of the BMU on the SOM grid.
Plotting the Marker:

plt.plot(w[0] + 0.5, w[1] + 0.5, markers[0], ...):
Plots a marker at the center of the BMU.
The +0.5 adjustment centers the marker inside each cell.
--------

8. Displaying the Plot


plt.title('Customer Segments using Self-Organizing Map')
plt.show()
--------
plt.title('...'): Adds a title to the plot.
plt.show(): Displays the final SOM visualization, showing clusters and boundaries between different segments.


Summary of Parameters:
Grid Size (x and y): Determines the SOM grid structure. Larger sizes are better for complex data.
Input Length (input_len): Set to match the number of features in the dataset.
Sigma (sigma): Controls the neighborhood size around each node.
Learning Rate (learning_rate): Sets the learning speed. Decreases over time for convergence.
Number of Iterations (num_iteration): Determines how many times the SOM will go through the training data.

