import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["hours_studied", "attendance"]]
y = data["final_score"]

# Model
model = LinearRegression()
model.fit(X, y)

# Predict
print(model.predict([[5, 90]]))  # example prediction
