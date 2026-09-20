import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("real_estate.csv") 

X = df[["area_in_sqft"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse)**0.5 
r2 = r2_score(y_test, y_pred)

print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)
print("\n---Evaluation Metrics---")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 : {r2:.2f}")


new_area = np.array([[3000]])
print("Predicted price for 3000 sqft: Rs.", model.predict(new_area)[0])


plt.figure(figsize=(7,5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line") 
plt.xlabel("Area in SQFT")
plt.ylabel("PRICE")
plt.title("AREA  vs PRICE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
