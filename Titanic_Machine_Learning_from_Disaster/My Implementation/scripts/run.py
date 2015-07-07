#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Abhijeet'

import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from scripts.populate_missing_ages import setMissingAges as sma
import re

# Reading training data and test data
train_df = pd.read_csv('data/train.csv', header=0)
test_df = pd.read_csv('data/test.csv', header=0)

# merge both data frames
df = pd.concat([train_df, test_df])

df.reset_index(inplace=True)

df.drop(labels='index', axis=1, inplace=True)

df = df.reindex_axis(train_df.columns, axis=1)

print df.shape[1], "columns:", df.columns.values
print "Row count:", df.shape[0]

df['Gender'] = df['Sex'].map({'female': 0, 'male':1})
#df['Port'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['AgeFill'] = df['Age']

# Missing values

# df['Fare'][np.isnan(df['Fare'])] = df['Fare'].median()
df.loc[(df.Fare.isnull()),'Fare'] = df['Fare'].median()

# Most occurring value
#df.loc[(df.Port.isnull()),'Port'] = 2

df = sma(df)

df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

# Assign dummy variable to cabin
df.loc[(df.Cabin.isnull()), 'Cabin'] = 'U0'
df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())
df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]


# StandardScaler will subtract the mean from each value then scale to the unit variance
scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['AgeFill'])

# Divide all fares into quartiles
df['Fare_bin'] = pd.qcut(df['Fare'], 4)

# qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# factorize or create dummies from the result
#df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])

df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)



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
