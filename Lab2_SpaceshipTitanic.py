import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.drop(['Cabin', 'Name'], axis=1, inplace=True)
test.drop(['Cabin', 'Name'], axis=1, inplace=True)

sns.countplot(x='Transported', data=train)
plt.show()

sns.histplot(train['Age'].dropna(), bins=30)
plt.show()

train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)

combined = pd.concat([train.drop('Transported', axis=1), test], axis=0)
combined = pd.get_dummies(combined)

X = combined.iloc[:len(train)]
X_test_final = combined.iloc[len(train):]
y = train['Transported'].astype(int)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

print("Validation Accuracy:", rf.score(X_val, y_val))

cv_scores = cross_val_score(rf, X, y, cv=5)
print("CV Accuracy:", np.mean(cv_scores))

preds = rf.predict(X_test_final)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': preds.astype(bool)
})

submission.to_csv("submission.csv", index=False)
print("Submission file created successfully!")
