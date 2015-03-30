__author__ = 'Abhijeet'


import pandas as pd
import numpy as np
import pylab as P
import csv
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/train.csv',header=0)

# df['Gender'] = df['Sex'].map(lambda x: 0 if x == "female" else 1)
df['Gender'] = df['Sex'].map({'female': 0, 'male':1})
df['Port'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['AgeFill'] = df['Age']

median_ages = np.zeros((2,3)) # 2 gender,3 classes
for i in range(0,2):
    for j in range(0,3):
        median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()


for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) &(df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

df.loc[(df.Port.isnull()),'Port'] = 2

# TODO: Feature Engineering
# Family Size
# df['FamilySize'] = df.SibSp + df.Parch

# age * Class
# df['Age*Class'] = df.AgeFill * df.Pclass

# Titles in Names, maybe higher Priority
# f = lambda x: 'Rev' in x
# df.Name.map(f)

# Child died? parents most probably die.

df = df.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
train_data = df.values



test_df = pd.read_csv('data/test.csv', header=0)

test_df['Gender'] = test_df['Sex'].map({'female': 0, 'male':1})
test_df['Port'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['AgeFill'] = test_df['Age']

median_ages = np.zeros((2,3)) # 2 gender,3 classes
for i in range(0,2):
    for j in range(0,3):
        median_ages[i, j] = test_df[(test_df['Gender'] == i) & (test_df['Pclass'] == j+1)]['Age'].dropna().median()

for i in range(0, 2):
    for j in range(0, 3):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Gender == i) &(test_df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

passengerid = test_df.PassengerId.values

test_df = test_df.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
test_df.loc[(test_df.Fare.isnull()) & (test_df.Pclass == 3), 'Fare'] = 12.45
test_data = test_df.values


forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
output = forest.predict(test_data)

prediction_file = open("data/output/random_forests.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

for i in range(len(passengerid)):
    prediction_file_object.writerow([passengerid[i], "%d" % int(output[i])])

prediction_file.close()
