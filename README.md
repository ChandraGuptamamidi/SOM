Dataset: Mall Customer Segmentation Data

The dataset includes information about customers, such as:
•	Customer ID
•	Gender
•	Age
•	Annual Income
•	Spending Score
Implementation Details
i. About the Dataset
The Mall Customer Segmentation dataset is used for customer clustering. Each data point represents a customer with features that include demographic data (age, gender) and behavioural attributes (annual income, spending score). This dataset is typically used to understand customer segments for targeted marketing.
•	Source: Kaggle,
•	Attributes:
1.	CustomerID: Unique identifier for each customer.
2.	Gender: Customer gender (Male/Female).
3.	Age: Age of the customer.
4.	Annual Income (k$): Customer’s annual income in thousands of dollars.
5.	Spending Score (1-100): Score based on customer behaviour and spending patterns.
ii. Read Me Document
•	Software Requirements:
o	Python 3.x
o	Libraries: NumPy, pandas, matplotlib, SciPy, scikit-learn, minisom (for SOM implementation)
•	Hardware Requirements:
o	Standard PC or Laptop
o	RAM: Minimum 4GB
•	Steps to Execute:
1.	Clone the repository from the provided GitHub link.
2.	Navigate to the folder in your terminal.
3.	Run pip install -r requirements.txt to install the required libraries.
4.	Execute the script using: python code.py.
iii. Executable File
The code.py script should include:
•	Data loading and preprocessing.
•	Applying a Self-Organizing Map using the minisom library.
•	Visualizing the clusters on a 2D map.
•	Highlighting different customer segments with distinct colours.

