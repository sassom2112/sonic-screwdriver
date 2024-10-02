#Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data from Excel
file_path = 'Module_1_Assignment_Spreadsheet.xlsx'
data = pd.read_excel(file_path)

# Extract X and Y
X = data['X'].values.reshape(-1,1) # Feature
Y = data['Y'].values.reshape(-1,1) # Target

# Normalize the data
X = (X - np.mean(X) / np.std(X))

# Implement Gradient Descent

# Hyperparamets
lr = 0.01         # Learning Rate
iterations = 1000

# Number of training examples
m = len(Y)

# Init wieghts and bais
theta = np.random.randn(1,1)
bias = 0

# Gradient Descent
for i in range(iterations):
  # Prediction
  Y_pred = np.dot(X, theta) + bias

  # Calculate the cost using the 'Mean Squared Error'
  cost = (1/(2*m)) * np.dot(X.T, (Y_pred -Y) ** 2)


  # Gradients 
  d_theta = (1/m) * np.dot(X.T, (Y_pred - Y))
  d_bias =  (1/m) * np.sum(Y_pred - Y)

  # Update parameters 
  theta -= lr * d_theta
  bias -= lr * d_bias

# Final parameters after gradient descent
print(f"Optimized Theta: {theta}")
print(f"Optimized Theta: {bias}")


# Plotting the data and the regression line
plt.scatter(X, Y, color='blue')
plt.plot(X, np.dot(X, theta) + bias, color='red')  # Regression line
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

