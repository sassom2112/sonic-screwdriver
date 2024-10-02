import pandas as pd
from sklearn.linear_model import LinearRegression


# Load Data
# Load data from Excel
file_path = 'Module_1_Assignment_Spreadsheet.xlsx'
data = pd.read_excel(file_path)

# Extract X and Y
X = data['X'].values.reshape(-1,1) # Feature
Y = data['Y'].values.reshape(-1,1) # Target

# Fit a linear regression model
model = LinearRegression()
model.fit(X,Y)

# Co-efficient (theta) and intercept (bias)
print(f"Optimized Theta: {model.coef_}")
print(f"Optimized Theta: {model.intercept_}")

