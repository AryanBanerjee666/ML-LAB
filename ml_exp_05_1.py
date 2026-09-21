import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

data={
    "StudyHours":[2, 4, 6, 8, 10, 12, 14, 3, 5, 7, 9, 11],
    "Attendance":[60, 65, 70, 75, 80, 85, 90, 55, 68, 72, 78, 88],
    "Pass":[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1] 
}
df= pd.DataFrame(data)

X= df[["StudyHours", "Attendance"]]
y= df["Pass"]

X_train,X_test,y_train, y_test= train_test_split( X,y,test_size=0.5,random_state=42,stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)

print("---Logistic Regression Performance---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:{precision_score(y_test, y_pred):.4f}")
print(f"Recall:{recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
