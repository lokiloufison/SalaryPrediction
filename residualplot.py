import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example dataset
data = pd.read_csv("C:/Users/satwi/Desktop/salary_dataset_part1.csv")
df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['YearsExperience']]  # Independent variable
y = df['Salary']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions for the test set
y_pred = model.predict(X_test)

# Check shapes
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of y_pred: {y_pred.shape}")

# Calculate residuals
residuals = y_test - y_pred

# Create the residual plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='purple', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Years of Experience')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True)
plt.show()