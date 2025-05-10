import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset

try:
    data = pd.read_csv("C:/Users/satwi/Desktop/salary_dataset_part1.csv")
    print(data.head())
except FileNotFoundError:
    print("Error: 'salary_dataset_part1.csv' not found. Ensure the file is in the correct directory.")
    exit()

# Check for missing values
print(data.isnull().sum())

# Scatter plot with trendline
try:
    figure = px.scatter(data_frame=data, 
                        x="Salary",
                        y="YearsExperience", 
                        size="YearsExperience", 
                        trendline="ols")
    figure.show()
except Exception as e:
    print(f"Error in plotting: {e}")

# Prepare data for training
try:
    x = np.asanyarray(data[["YearsExperience"]])
    y = np.asanyarray(data[["Salary"]])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
    model = LinearRegression()
    model.fit(xtrain, ytrain)
except Exception as e:
    print(f"Error in model training: {e}")

# Predict salary based on user input
try:
    a = float(input("Years of Experience: "))
    features = np.array([[a]])
    predicted_salary = model.predict(features)
    print("Predicted Salary = ", predicted_salary[0][0])
except ValueError:
    print("Invalid input. Please enter a numeric value for years of experience.")
except Exception as e:
    print(f"Error in prediction: {e}")