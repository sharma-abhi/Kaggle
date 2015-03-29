__author__ = 'Abhijeet'

import pandas as pd
import numpy as np
import pylab as P

df = pd.read_csv('data/train.csv',header=0)

df.Age.dropna().hist(bins = 16, range=(0, 80), alpha= .5)

# calculating median ages per gender per classes
median_ages = np.zeros((2,3)) # 2 gender,3 classes
for i in range(0,2):
    for j in range(0,3):
        median_ages[i, j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()


for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) &(df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

# Feature Engineering
df['FamilySize'] = df.SibSp + df.Parch

df['Age*Class'] = df.AgeFill * df.Pclass

# df.Pclass.value_counts()
# f = lambda x: 'Mr' in x
# df.Name.map(f)

# df.dtypes[df.dtypes.map(lambda x:x == 'object')]
# df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
# df = df.drop(['Age'], axis=1)
# df = df.dropna()

train_data = df.values