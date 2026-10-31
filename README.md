# SGD Classifier

## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris dataset into a DataFrame.
2. Split the data into features (X) and target (species), then into training and testing sets.
3. Create an SGDClassifier and train it on the training data.
4. Predict the species for the test data and evaluate the model using accuracy and the confusion matrix.

## Program:
```python
'''
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Sushil Shiva R
RegisterNumber: 212224250017
'''
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)
print("Predicted:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
```

## Output:
<img width="752" height="367" alt="image" src="https://github.com/user-attachments/assets/a10291e7-d8cf-4f5e-b7c8-cf2730b320f5" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
 
