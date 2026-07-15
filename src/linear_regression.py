import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/wine.csv")

X = df[["alcohol", "volatile acidity", "sulphates"]]

y = df["quality"]

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred)
plt.xlabel("Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression")
plt.grid(True)

plt.savefig("linear_regression_result.png")

plt.show()

