import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "area_in_sqft": [1500, 2000, 2500, 3000, 3500],
    "bedrooms": [3, 4, 3, 5, 4],
    "price": [3000000, 4000000, 4500000, 6000000, 6500000]
}

df = pd.DataFrame(data)

X = df[["area_in_sqft", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

new_house = np.array([[2200, 3]])
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])
