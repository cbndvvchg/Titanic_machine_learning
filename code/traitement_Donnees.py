import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


"""import des datasets"""

test_set = pd.read_csv("data/test.csv")

train_set = pd.read_csv("data/train.csv")

gender_set = pd.read_csv("data/gender_submission.csv")


X = train_set.drop("Survived", axis=1)
y = train_set["Survived"]

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestClassifier(n_estimators=100, random_state=42)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])





X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy on the validation set: {accuracy:.2f}")


test_predictions = clf.predict(test_set)

submission = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': test_predictions})
submission.to_csv('submission.csv', index=False)



