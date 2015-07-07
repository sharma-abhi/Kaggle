__author__ = 'Abhijeet'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

### Populate missing ages  using RandomForestClassifier
def setMissingAges(df):

    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['AgeFill', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Gender', 'Port']]

    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[(df.AgeFill.notnull()) ]
    unknownAge = age_df.loc[(df.AgeFill.isnull()) ]

    # All age values are stored in a target array
    y = knownAge.values[:, 0]

    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[(df.AgeFill.isnull()), 'AgeFill'] = predictedAges

    return df