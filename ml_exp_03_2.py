import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = {
    "Age": [20, 21, None, 23, 24],
    "Salary": [25000, 80000, 28000, None, 75000],
    "Department": ["IT", "CSE", None, "ECE", "IT"],
    "Years_of_Experience": [5, 10, 1, None, 12]
}
df = pd.DataFrame(data)

X = df.drop("Salary", axis=1)
y = df["Salary"]

numeric_features = ["Age", "Years_of_Experience"]
categorical_features = ["Department"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Original Dataset ---")
print(df)

print("\n--- Processed Features Matrix ---")
print(X_processed)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
