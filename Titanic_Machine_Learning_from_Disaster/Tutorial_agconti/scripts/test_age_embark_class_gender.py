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
import statsmodels.api as sm
import predict as ka
from patsy import dmatrices


df = pd.read_csv("../data/train.csv") 
df = df.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
df = df.dropna() 

# here the ~ sign is an = sign, and the features of our dataset
# are written as a formula to predict survived. The C() lets our 
# regression know that those variables are categorical.
formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' 

# create a results dictionary to hold our regression results for easy analysis later        
results = {} 

# Create a regression friendly version of our data using dmatrices
y,x = dmatrices(formula, data=df, return_type='dataframe')

# Create a Logit Model Based on our data
model=sm.Logit(y,x)

# Fit that Model to the Data
res = model.fit()
results['Logit'] = [res, formula]

#results.params
res.summary()
###################################################################################################

# The file is already trained on the train.csv file. 
# Now we test our model by making predictions on the test.csv file. 
# You'll notice the test.csv file has no values in the survived field. This
# Is what we're are tying to predict. 

test_data = pd.read_csv("../data/test.csv") 
t2 = [i for i in test_data['PassengerId']]
test_data = test_data.drop(['Ticket','Cabin'], axis=1)
# Remove NaN values
test_data = test_data.dropna() 
test_data['Survived'] = 1.23
predicted_results = ka.predict(test_data, results, 'Logit')
predicted_results  = Series(predicted_results) 

d = {}
t1 = [i for i in test_data['PassengerId']]
p = [i for i in predicted_results]
for i in range(len(t1)):
	if p[i] > 0.5:
		d[t1[i]] = 1
	else:
		d[t1[i]] = 0
	#d[t1[i]] = p[i]

for j in t2:
	if j not in t1:
		d[j] = 0

df2 = pd.DataFrame(d.items())

df2.to_csv("../data/output/logitregres.csv")  

print "Analysis ended"
