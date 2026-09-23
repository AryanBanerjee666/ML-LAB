import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.datasets import make_classification

X, y = make_classification( n_samples=20,n_features=5, n_informative=3, n_redundant=0,random_state=42)
print("Feature Matrix X:")
print(pd.DataFrame(X))
print("Target Vector y:")
print(y) 

X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.5, random_state=42,stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:,1]

def evaluate_threshold(threshold):
    y_pred = (y_probs >= threshold).astype(int)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"Threshold = {threshold} ")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")

for t in [0.3, 0.5, 0.7]:
    evaluate_threshold(t)
