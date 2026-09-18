import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

data = {
    "Age": [20, 21, None, 23, 24],
    "Salary":[50000,75000,None,80000,30000],
    "Years_of_Experience": [5, 10, 1, None, 12]
}
df = pd.DataFrame(data)

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df)
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)

print("--- Original Data ---")
print(X)

print("\n--- StandardScaler Output ---")
print(X_std)

print("\n--- MinMaxScaler Output ---")
print(X_mm)
