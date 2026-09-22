import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df= pd.DataFrame({
    "area":[1500, 2000, 2500, 3000, 3500],
    "bedrooms":[3, 4, 3, 5, 4],
    "price": [3000000, 4000000, 4500000, 6000000, 6500000]
})
X = df[["area", "bedrooms"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lin=linear_model.predict(X_test)
linear_r2 = r2_score(y_test,y_pred_lin)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly, y_train)
y_pred_poly = polynomial_model.predict(X_test_poly)
polynomial_r2 = r2_score(y_test,y_pred_poly)

print(f"Linear Regression R2: {linear_r2:.4f}")
print(f"Polynomial Regression R2: {polynomial_r2:.4f}")

