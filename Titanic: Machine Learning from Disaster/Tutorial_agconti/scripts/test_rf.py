# A model for prediction survival on the Titanic based on where an 
# individual Embarked, their gender, or the class they traveled in. 
# AGC 2013
# 
# 
# Here Will will run generate predictions of who survived and who did not
# from our basic Least Squares Regression model.
# Our Formula is :
# survived_prediction = C + pclass + sex + age + sibsp  + embarked

# Import Utilities
import csv as csv
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#import statsmodels.api as sm
#import predict as ka
from patsy import dmatrices
from sklearn import datasets, svm
import sklearn.ensemble as ske


df = pd.read_csv("../data/train.csv") 
df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
df = df.dropna() 

# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our 
# regression know that those variables are categorical.
formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + Parch + C(Embarked)' 

# create a results dictionary to hold our regression results for easy analysis later        
results = {} 

# Create the random forest model and fit the model to our training data
y, x = dmatrices(formula_ml, data=df, return_type='dataframe')
# RandomForestClassifier expects a 1 demensional NumPy array, so we convert
y = np.asarray(y).ravel()

#instantiate and fit our model
results_rf = ske.RandomForestClassifier(n_estimators=100).fit(x, y)
# Score the results
score = results_rf.score(x, y)
print "Mean accuracy of Random Forest Predictions on the data was: {0}".format(score)



print "Analysis ended"
