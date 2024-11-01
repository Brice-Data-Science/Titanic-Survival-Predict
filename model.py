"""
Model.

Spyder Editor.

Created on Sat Oct 26 01:53:52 2024
@author: BRICE NELSON

Machine learning file.  This file will create logic to predict survival of
passengers on the Titanic.
"""
import os
# import numpy as np # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

# This Python 3 environment comes with many helpful analytics libraries
# installed
# It is defined by the kaggle/python Docker image:
#   https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list
# all files under the input directory


for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

try:
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women / len(women))
print(f'Percent of women who survived: {rate_women}')

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men) / len(men)
print(f"Percent of men who survived: {rate_men}")


y = train_data["Survived"]

features = ["Pclass", 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.to_csv('results/submission.csv', index=False)
print("Your submission was successfully saved")
